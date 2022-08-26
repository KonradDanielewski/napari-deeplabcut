from collections import defaultdict
import gc
import os
from types import MethodType
from typing import Optional, Sequence, Union

import napari
import numpy as np
import pandas as pd

import re

from napari.layers import Image, Points, Shapes
from napari.layers.points._points_key_bindings import register_points_action
from napari.layers.utils import color_manager
from napari.utils.events import Event
from napari.utils.history import get_save_history, update_save_history
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_deeplabcut import keypoints
from napari_deeplabcut.misc import to_os_dir_sep


def _get_and_try_preferred_reader(
    self,
    dialog,
    *args,
):
    try:
        self.viewer.open(
            dialog._current_file,
            plugin="napari-deeplabcut",
        )
    except ValueError:
        self.viewer.open(
            dialog._current_file,
            plugin="builtins",
        )


# Hack to avoid napari's silly variable type guess,
# where property is understood as continuous if
# there are more than 16 unique categories...
def guess_continuous(property):
    if issubclass(property.dtype.type, np.floating):
        return True
    else:
        return False


color_manager.guess_continuous = guess_continuous


# Hack to save a KeyPoints layer without showing the Save dialog
def _save_layers_dialog(self, selected=False):
    """Save layers (all or selected) to disk, using ``LayerList.save()``.
    Parameters
    ----------
    selected : bool
        If True, only layers that are selected in the viewer will be saved.
        By default, all layers are saved.
    """
    selected_layers = list(self.viewer.layers.selection)
    msg = ""
    if not len(self.viewer.layers):
        msg = "There are no layers in the viewer to save."
    elif selected and not len(selected_layers):
        msg = (
            "Please select one or more layers to save," '\nor use "Save all layers..."'
        )
    if msg:
        QMessageBox.warning(self, "Nothing to save", msg, QMessageBox.Ok)
        return
    if len(selected_layers) == 1 and isinstance(selected_layers[0], Points):
        self.viewer.layers.save("", selected=True, plugin="napari-deeplabcut")
        self.viewer.status = "Data successfully saved"
    else:
        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)
        filename, _ = dlg.getSaveFileName(
            caption=f'Save {"selected" if selected else "all"} layers',
            dir=hist[0],  # home dir by default
        )
        if filename:
            self.viewer.layers.save(filename, selected=selected)


class KeypointControls(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.inserted.connect(self.on_insert)
        self.viewer.layers.events.removed.connect(self.on_remove)

        self.viewer.window.qt_viewer._get_and_try_preferred_reader = MethodType(
            _get_and_try_preferred_reader,
            self.viewer.window.qt_viewer,
        )

        self._label_mode = keypoints.LabelMode.default()

        # Hold references to the KeypointStores
        self._stores = {}

        # Storage for extra image metadata that are relevant to other layers.
        # These are updated anytime images are added to the Viewer
        # and passed on to the other layers upon creation.
        self._images_meta = dict()

        # Add some more controls
        self._layout = QVBoxLayout(self)
        self._menus = []
        self._radio_group = self._form_mode_radio_buttons()

        # Substitute default menu action with custom one
        for action in self.viewer.window.file_menu.actions():
            if "save selected layer" in action.text().lower():
                action.triggered.disconnect()
                action.triggered.connect(
                    lambda: _save_layers_dialog(
                        self.viewer.window.qt_viewer,
                        selected=True,
                    )
                )
                break

        self._multiview_controls = MultiViewControls(self)
        self._layout.addWidget(self._multiview_controls)

    def _form_dropdown_menus(self, store):
        menu = KeypointsDropdownMenu(store)
        self._menus.append(menu)
        layout = QVBoxLayout()
        layout.addWidget(menu)
        self._layout.addLayout(layout)

    def _form_mode_radio_buttons(self):
        layout1 = QVBoxLayout()
        title = QLabel("Labeling mode")
        layout1.addWidget(title)
        layout2 = QHBoxLayout()
        group = QButtonGroup(self)

        for i, mode in enumerate(keypoints.LabelMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            btn.setToolTip(keypoints.TOOLTIPS[mode])
            group.addButton(btn, i)
            layout2.addWidget(btn)
        group.button(1).setChecked(True)
        layout1.addLayout(layout2)
        self._layout.addLayout(layout1)

        def _func():
            self.label_mode = group.checkedButton().text()

        group.buttonClicked.connect(_func)
        return group

    def _remap_frame_indices(self, layer):
        if "paths" not in self._images_meta:
            return

        new_paths = [to_os_dir_sep(p) for p in self._images_meta["paths"]]
        paths = layer.metadata.get("paths")
        if paths is not None and np.any(layer.data):
            paths_map = dict(zip(range(len(paths)), map(to_os_dir_sep, paths)))
            # Discard data if there are missing frames
            missing = [i for i, path in paths_map.items() if path not in new_paths]
            if missing:
                if isinstance(layer.data, list):
                    inds_to_remove = [
                        i
                        for i, verts in enumerate(layer.data)
                        if verts[0, 0] in missing
                    ]
                else:
                    inds_to_remove = np.flatnonzero(np.isin(layer.data[:, 0], missing))
                layer.selected_data = inds_to_remove
                layer.remove_selected()
                for i in missing:
                    paths_map.pop(i)

            # Check now whether there are new frames
            temp = {k: new_paths.index(v) for k, v in paths_map.items()}
            data = layer.data
            if isinstance(data, list):
                for verts in data:
                    verts[:, 0] = np.vectorize(temp.get)(verts[:, 0])
            else:
                data[:, 0] = np.vectorize(temp.get)(data[:, 0])
            layer.data = data
        layer.metadata.update(self._images_meta)

    def on_insert(self, event):
        layer = event.source[-1]
            
        if isinstance(layer, Image):
            paths = layer.metadata.get("paths")
            if paths is None:
                return
            # Store the metadata and pass them on to the other layers
            self._images_meta.update(
                {
                    "paths": paths,
                    "shape": layer.level_shapes[0],
                    "root": layer.metadata["root"],
                }
            )
            # FIXME Ensure the images are always underneath the other layers
            # self.viewer.layers.selection = []
            # if (ind := event.index) != 0:
            #     order = list(range(len(self.viewer.layers)))
            #     order.remove(ind)
            #     new_order = [ind] + order
            #     self.viewer.layers.move_multiple(new_order)
            # if (ind := event.index) != 0:
            #     self.viewer.layers.move_selected(ind, 0)
        elif isinstance(layer, Points):
            store = keypoints.KeypointStore(self.viewer, layer)
            self._stores[layer] = store
            # TODO Set default dir of the save file dialog
            if root := layer.metadata.get("root"):
                update_save_history(root)
            layer.metadata["controls"] = self
            layer.text.visible = False
            layer.bind_key("M", self.cycle_through_label_modes)
            layer.add = MethodType(keypoints._add, store)
            layer.events.add(query_next_frame=Event)
            layer.events.query_next_frame.connect(store._advance_step)
            layer.bind_key("Shift-Right", store._find_first_unlabeled_frame)
            layer.bind_key("Shift-Left", store._find_first_unlabeled_frame)
            self.viewer.dims.events.current_step.connect(
                store.smart_reset,
                position="last",
            )
            store.smart_reset(event=None)
            layer.bind_key("Down", store.next_keypoint, overwrite=True)
            layer.bind_key("Up", store.prev_keypoint, overwrite=True)
            layer.face_color_mode = "cycle"
            layer.events.data.connect(self.update_data)
            if not self._menus:
                self._form_dropdown_menus(store)
            # adding layer for epipolar lines
            # TODO this should only happen when more than one camera
            # so should be moved somewhere else

            # 0, 0, 3 seems to mean that the layer is 3 dimensional (x,y,framenumber), 0,0,2 makes a 2 dimensional layer
            self.help_layer = self.viewer.add_shapes(np.empty((0, 0, 3)), name="Help lines")
                        
            # attaching extrinsic calibration to KeypointControls
            if self.viewer.title.__contains__("Camera "):
                self.help_layer.metadata["viewers"] = self._multiview_controls._viewers
                self.help_layer.metadata["camera_number"] = int(re.findall(r'\d+',self.viewer.title)[-1])
                self.help_layer.metadata["calibration_type"] ="unknown"
                self.help_layer.metadata["extrinsic_calibration_coefficients"] = []
                self.help_layer.metadata["point_layer"] = layer

                for file in os.listdir(layer.metadata["root"]):
                    if file.__contains__("dltCoefs.csv"):
                        self.help_layer.metadata["calibration_type"] = "DLTdv"
                        self.help_layer.metadata["extrinsic_calibration_coefficients"] = pd.read_csv(
                            os.path.join(layer.metadata["root"],
                            file), header=None).values[:, self.help_layer.metadata["camera_number"]-1]

        for layer_ in self.viewer.layers:
            if not isinstance(layer_, Image):
                self._remap_frame_indices(layer_)

    def update_data(self, event):
        self._multiview_controls._update_viewers_data(event.value, event.source)
        #self._multiview_controls._update_viewers_data(event.value, event.source, self.viewer)

    def on_remove(self, event):
        layer = event.value
        if isinstance(layer, Points):
            self._stores.pop(layer, None)
            while self._menus:
                menu = self._menus.pop()
                self._layout.removeWidget(menu)
                menu.setParent(None)
                menu.destroy()
        elif isinstance(layer, Image):
            self._images_meta = dict()

    def on_move(self, event):
        print("moved")

    @register_points_action("Change labeling mode")
    def cycle_through_label_modes(self, *args):
        self.label_mode = next(keypoints.LabelMode)

    @property
    def label_mode(self):
        return str(self._label_mode)

    @label_mode.setter
    def label_mode(self, mode: Union[str, keypoints.LabelMode]):
        self._label_mode = keypoints.LabelMode(mode)
        self.viewer.status = self.label_mode
        for btn in self._radio_group.buttons():
            if btn.text() == str(mode):
                btn.setChecked(True)
                break


@Points.bind_key("F")
def toggle_face_color(layer):
    if layer._face.color_properties.name == "id":
        layer.face_color = "label"
        layer.face_color_cycle = layer.metadata["face_color_cycles"]["label"]
    else:
        layer.face_color = "id"
        layer.face_color_cycle = layer.metadata["face_color_cycles"]["id"]
    layer.events.face_color()


@Points.bind_key("E")
def toggle_edge_color(layer):
    # Trick to toggle between 0 and 2
    layer.edge_width = np.bitwise_xor(layer.edge_width, 2)


class DropdownMenu(QComboBox):
    def __init__(self, labels: Sequence[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.addItems(labels)

    def update_to(self, text: str):
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def reset(self):
        self.setCurrentIndex(0)


class KeypointsDropdownMenu(QWidget):
    def __init__(
        self,
        store: keypoints.KeypointStore,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.store = store
        self.store.layer.events.current_properties.connect(self.update_menus)

        # Map individuals to their respective bodyparts
        self.id2label = defaultdict(list)
        for keypoint in store._keypoints:
            label = keypoint.label
            id_ = keypoint.id
            if label not in self.id2label[id_]:
                self.id2label[id_].append(label)

        self.menus = dict()
        if store.ids[0]:
            menu = create_dropdown_menu(store, list(self.id2label), "id")
            menu.currentTextChanged.connect(self.refresh_label_menu)
            self.menus["id"] = menu
        self.menus["label"] = create_dropdown_menu(
            store, self.id2label[store.ids[0]], "label"
        )
        layout = QVBoxLayout()
        title = QLabel("Keypoint selection")
        layout.addWidget(title)
        for menu in self.menus.values():
            layout.addWidget(menu)
        layout.addStretch(1)
        self.setLayout(layout)

    def update_menus(self, event):
        keypoint = self.store.current_keypoint
        for attr, menu in self.menus.items():
            val = getattr(keypoint, attr)
            if menu.currentText() != val:
                menu.update_to(val)

    def refresh_label_menu(self, text: str):
        menu = self.menus["label"]
        menu.blockSignals(True)
        menu.clear()
        menu.addItems(self.id2label[text])
        menu.blockSignals(False)


def create_dropdown_menu(store, items, attr):
    menu = DropdownMenu(items)

    def item_changed(ind):
        current_item = menu.itemText(ind)
        if current_item is not None:
            setattr(store, f"current_{attr}", current_item)

    menu.currentIndexChanged.connect(item_changed)
    return menu


class MultiViewControls(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)
        self._viewers = []
        label = QLabel("# of views")
        self._box = QSpinBox()
        self._box.setMinimum(2)
        self._box.setMaximum(4)
        layout.addWidget(label)
        layout.addWidget(self._box)
        button = QPushButton("Open Viewers")
        button.clicked.connect(self._open_viewers)
        epl_button = QPushButton("Draw ep lines")
        epl_button.clicked.connect(self._initiate_ep_lines)
        layout.addWidget(button)
        layout.addWidget(epl_button)
        self.setLayout(layout)

        # Grid-like positioning in screen
        self._screen_size = QGuiApplication.primaryScreen().size()
        self._half_w = self._screen_size.width() // 2
        self._half_h = self._screen_size.height() // 2
        self._pos = [
            (0, 0),
            (self._half_w, 0),
            (0, self._half_h),
            (self._half_w, self._half_h),
        ]

    def _open_viewers(self, *args):
        n = self._box.value()
        self.parent.viewer.title = "Camera 1"
        self._viewers.append(self.parent.viewer)
        for i in range(n - 1):
            viewer = napari.Viewer(title=f"Camera {i + 2}")
            # Auto activate plugin
            for action in viewer.window.plugins_menu.actions():
                if str(action.text()).__contains__("napari-deeplabcut"):
                    action.trigger()
            self._viewers.append(viewer)
        for n, viewer in enumerate(self._viewers):
            viewer.window.resize(self._half_w, self._half_h)
            viewer.window._qt_window.move(*self._pos[n % 4])
            viewer.dims.events.current_step.connect(self._update_viewers)
            #viewer.dims.events.current_step.connect(self._update_viewers)
        # dirty hack to get the MultiViewControls of the newly launched viewers
        for object in gc.get_objects():
            if isinstance(object, MultiViewControls):
                object._viewers = self._viewers

    def _update_viewers(self, event):
        ind = event.value[0]
        for viewer in self._viewers:
            viewer.dims.set_current_step(0, ind)
        self._update_viewers_data(event.value, event.source)

    def _initiate_ep_lines(self):
        print("Initiating epipolar lines.")
        viewers = self._viewers
        current_viewer = self.parent.viewer
        other_viewers = [x for x in viewers if not x==current_viewer]
        
        number_of_frames = current_viewer.dims.range[0][1]
        number_of_other_cams = len(other_viewers)

        for layer in current_viewer.layers:
            if layer.name.__contains__("CollectedData"):
                point_layer2 = layer
            elif layer.name.__contains__("Help lines"):
                help_layer2 = layer
            elif layer.name.__contains__("images"):
                image_layer2 = layer
        
            frame_width2 = image_layer2.data.shape[2]
            frame_height2 = image_layer2.data.shape[1]

        bodypart_labels = self.parent._stores[point_layer2].labels
        number_of_bodyparts = len(bodypart_labels)

        #all_lines=np.zeros(number_of_other_cams,number_of_bodyparts,number_of_frames,4)
        #layer.add(np.array([[imgy1, imgx1], [imgy2, imgx2]]), shape_type='line', edge_color=[0,1,0],edge_width=3)


        print(number_of_frames)
        print(type(number_of_frames))

        C2 = help_layer2.metadata["extrinsic_calibration_coefficients"]
        lines_to_add=[]
        line_colors=[]
        current_frame_number = current_viewer.dims.current_step[0]
        for viewer in other_viewers:
            print("Now doing:", viewer.title)
            for layer in viewer.layers:
                if layer.name.__contains__("CollectedData"):
                    point_layer = layer
                elif layer.name.__contains__("Help lines"):
                    help_layer = layer
            C1 = help_layer.metadata["extrinsic_calibration_coefficients"]
        
            for frame_number in range(int(number_of_frames)):
                viewer.dims.set_current_step(0,frame_number)
                current_colors = point_layer.face_color[[int(x)==frame_number for x in point_layer.data[:,0]]]
                current_data = point_layer.data[[int(x)==frame_number for x in point_layer.data[:,0]]]
                annotated_keypoints = [x[0] for x in point_layer.metadata['controls']._stores[point_layer].annotated_keypoints]
                for i,label in enumerate(bodypart_labels):
                    try:
                        label_index = annotated_keypoints.index(label)
                        [v1,u1] = current_data[label_index,[1,2]]
                        # uv = xy image coord but order is wrong here because
                        # convention for image data is wrong (and evil)
                        lines_to_add.append(get_epipolar_line_shape(u1,v1,C1,C2, frame_width2, frame_height2,frame_number))
                        line_colors.append(current_colors[label_index][0:3])

                    # if label not found among annotated keypoints, do this:
                    except:
                        lines_to_add.append(np.array([[frame_number,0,0],[frame_number,0,0]]))
                        line_colors.append([0,0,0])
        print(lines_to_add)
        viewer.dims.set_current_step(0,current_frame_number)
        help_layer2.add(lines_to_add, shape_type='line',edge_color=line_colors,edge_width=1)

            #for i, viewer in enumerate(viewers):
            #    for j, viewer2 in enumerate(viewers):
            #        if j == i: continue
            #        print("First viewer:", viewer.title)
            #        print("Second viewer:", viewer2.title)
            #        print("**********")





        # TODO Need to update layer properties as well
    def _update_viewers_data(self, data, source):
        #for i, viewer in enumerate(self._viewers):
        #    for j, viewer2 in enumerate(self._viewers):
        #        if j == i: continue
        #        try:
        #            draw_epipolars(viewer, viewer2)
        #        except:
        #            return
        #return

        #if len(self._viewers)>1:
        #    for viewer in self._viewers:
        #        for viewer2 in self._viewers:
        #            if viewer.title != viewer2.title and len(viewer.layers) > 2 and len(viewer2.layers) > 2:
        #                print("")
        #                draw_epipolars(viewer, viewer2)
        #return

        current_viewer = self.parent.viewer
        current_frame = current_viewer.dims.current_step[0]
        try:
            selected_point = list(source._selected_data)[0]
        except:
            return

        v1 = data[selected_point][1] # image y coord
        u1 = data[selected_point][2] # image x coord

        print("current viewer:", current_viewer.title)
        print("current frame:", current_frame)
        print("selected point:", selected_point)

        for layer in current_viewer.layers:
            if layer.name.__contains__("CollectedData"):
                point_layer = layer
            elif layer.name.__contains__("Help lines"):
                help_layer = layer
        C1 = help_layer.metadata["extrinsic_calibration_coefficients"]
        bodypart_labels = point_layer.metadata['controls']._stores[point_layer].labels
        annotated_keypoints = [x[0] for x in point_layer.metadata['controls']._stores[point_layer].annotated_keypoints]
        current_keypoint = annotated_keypoints[selected_point-np.argmax(point_layer.data[:,0]==current_frame)]
        print("Current bodypart:", current_keypoint)
        print("Bodypart index:",bodypart_labels.index(current_keypoint))
        print("Bodypart total index:",current_frame*len(bodypart_labels)+bodypart_labels.index(current_keypoint))
        line_index = current_frame*len(bodypart_labels)+bodypart_labels.index(current_keypoint)
        viewers = self._viewers
        other_viewers = [x for x in viewers if not x==current_viewer]
        number_of_frames = current_viewer.dims.range[0][1]
        number_of_other_cams = len(other_viewers)
        number_of_bodyparts = len(bodypart_labels)
        print("number_of_frames",number_of_frames)
        print("number_of_other_cams",number_of_other_cams)
        print("number_of_bodyparts",number_of_bodyparts)
        print("helplines len",len(help_layer.data))

        for i,viewer in enumerate(other_viewers):
            print(viewer.title)
            
            for layer in viewer.layers:
                print(layer.name)
                print(len(viewer.layers))
                if layer.name.__contains__("CollectedData"):
                    point_layer2 = layer
                elif layer.name.__contains__("Help lines"):
                    help_layer2 = layer
                elif layer.name.__contains__("images"):
                    image_layer2 = layer
                
            frame_width2 = image_layer2.data.shape[2]
            frame_height2 = image_layer2.data.shape[1]
            C2 = help_layer2.metadata["extrinsic_calibration_coefficients"]
            
            print("current data at", line_index)
            print(help_layer2.data[line_index])
            print("proposed replacement:")
            print(get_epipolar_line_shape(u1,v1,C1,C2, frame_width2, frame_height2,current_frame))
            data_temp = help_layer2.data.copy()
            data_temp[int(i*number_of_bodyparts*number_of_frames+line_index)] = get_epipolar_line_shape(u1,v1,C1,C2, frame_width2, frame_height2,current_frame)
            help_layer2.data=data_temp




        return
        v = data[selected_point][1] # image y coord
        u = data[selected_point][2] # image x coord

        C1 = self.parent.help_layer.metadata["extrinsic_calibration_coefficients"]

        imgx1=0
        imgx2=2560
        # no longer needed since we store viewers in MultiViewControls._viewers
        #viewers = self.parent.help_layer.metadata["viewers"]
        viewers = self._viewers
        
        for viewer in viewers:
            if viewer.title == "Camera 1":
                viewer1 = viewer
            elif viewer.title == "Camera 2":
                viewer2 = viewer
        draw_epipolars(viewer1, viewer2)
        return

        for viewer in viewers:
            print(viewer.title)

        for viewer in viewers:
            if viewer.title != current_viewer.title:
                for layer in viewer.layers:
                    if isinstance(layer, Shapes):
                        C2 = layer.metadata["extrinsic_calibration_coefficients"]
                        (m,b) = get_epipolar_line(u,v,C1,C2)
                        imgy1=(m*imgx1+b)
                        imgy2=(m*imgx2+b)
                        with layer.events.data.blocker():
                            layer.add(np.array([[imgy1, imgx1], [imgy2, imgx2]]), shape_type='line', edge_color=[0,1,0],edge_width=3)
                #for layer in viewer.layers:
                #    if isinstance(layer, Points):
                #        with layer.events.data.blocker():
                #            layer.data[selected_point] = data[selected_point]
                #            layer.refresh()

def invdlt(A,XYZ):

    X=XYZ[0]
    Y=XYZ[1]
    Z=XYZ[2]

    x=np.divide((X*A[0]+Y*A[1]+Z*A[2]+A[3]),(X*A[8]+Y*A[9]+Z*A[10]+1))
    y=np.divide((X*A[4]+Y*A[5]+Z*A[6]+A[7]),(X*A[8]+Y*A[9]+Z*A[10]+1))

    return x, y

def get_epipolar_line(u1,v1,C1,C2):
    '''This function is based on the partialdlt which is a MATLAB function
    written by Ty Hedrick and included with DLTdv (https://github.com/tlhedrick/dltdv)
    
    It takes the image coordinates for a point in View 1 (u,v) and the extrinsic calibration
    coefficients from both cameras/views (C1 and C2) as input.

    Note that u is the horizontal coordinate (x) and v the vertical (y)

    It then calculates an epipolar line for View 2 as a straight line with slope m and
    Y-intercept b.

    '''
    
    z=[500,-500] # Two random z values

    #pre-alocate x and y
    y = [0, 0] # could do np.zeros(2).tolist() or withour tolist but I don't know what that would mess up
    x = [0, 0]

    # calculate the x and y coordinates in real world coordinates
    for i in [0, 1]:
        Z = z[i]
        
        # Hedricks cites MathCAD 11 for the calculation of y and x
        y[i] =\
            -(u1*C1[8]*C1[6]*Z + u1*C1[8]*C1[7] - u1*C1[10]*Z*C1[4] -u1*C1[4] + C1[0]*v1*C1[10]*Z\
            + C1[0]*v1 - C1[0]*C1[6]*Z - C1[0]*C1[7] - C1[2]*Z*v1*C1[8] + C1[2]*Z*C1[4] - \
            C1[3]*v1*C1[8] + C1[3]*C1[4]) / (u1*C1[8]*C1[5] - u1*C1[9]*C1[4] + C1[0]*v1*C1[9] - \
            C1[0]*C1[5] - C1[1]*v1*C1[8] + C1[1]*C1[4])

        Y=y[i]

        x[i] = -(v1*C1[9]*Y+v1*C1[10]*Z+v1-C1[5]*Y-C1[6]*Z-C1[7])/(v1*C1[8]-C1[4])

    # Calculate two image coordinates for View 2 on which to base the straight line
    u2=[0,0]
    v2=[0,0]
    for i in [0,1]:
        (u2[i],v2[i]) = invdlt(C2,np.array([x[i],y[i],z[i]]))
    
    # Calculate slope and Y-intercept (m and b)
    m=(v2[1]-v2[0])/(u2[1]-u2[0])
    b=v2[0]-m*u2[0]

    return m,b

def draw_epipolars(viewer1, viewer2):
    '''
    Draw epipolars on viewer 2 based on the point data in viewer 1
    '''

    for layer in viewer1.layers:
                    if isinstance(layer, Shapes):
                        help_layer1 = layer
                    elif isinstance(layer,Points):
                        points_layer1 = layer
    for layer in viewer2.layers:
                    if isinstance(layer, Shapes):
                        help_layer2 = layer
                    elif isinstance(layer,Image):
                        image_layer2 = layer

    frame_width2 = image_layer2.data.shape[2]
    frame_height2 = image_layer2.data.shape[1]

    C1 = help_layer1.metadata["extrinsic_calibration_coefficients"]
    C2 = help_layer2.metadata["extrinsic_calibration_coefficients"]

    current_frame = viewer1.dims.current_step[0]
    current_point_indexes1 = points_layer1.data[:,0] == current_frame
    current_points1 = points_layer1.data[current_point_indexes1,:][:,[1,2]]
    current_face_colors = points_layer1.face_color[current_point_indexes1,:][:,0:3]

    
    #help_layer2.data = np.empty((0, 0, 2))
    for i, point in enumerate(current_points1):

        (m,b) = get_epipolar_line(point[1],point[0],C1,C2)
        imgx1 = 0
        imgx2 = frame_width2             
        imgy1=b
        imgy2=m*frame_width2+b

        if imgy1<0:
            imgy1 = 0
            imgx1 = -b/m
        elif imgy1>frame_height2:
            imgy1=frame_height2
            imgx1=(imgy1-b)/m
        
        if imgy2<0:
            imgy2 = 0
            imgx2 = -b/m
        elif imgy2>frame_height2:
            imgy2=frame_height2
            imgx2=(imgy2-b)/m
        
        line_color = current_face_colors[i]
        with help_layer2.events.data.blocker():
            help_layer2.add(np.array([[imgy1, imgx1], [imgy2, imgx2]]), shape_type='line', edge_color=line_color,edge_width=3)

def get_epipolar_line_shape(u1,v1,C1,C2, frame_width, frame_height, frame_number):
    
    (m,b) = get_epipolar_line(u1,v1,C1,C2)
    imgx1 = 0
    imgx2 = frame_width
    imgy1=b
    imgy2=m*frame_width+b
    if imgy1<0:
        imgy1 = 0
        imgx1 = -b/m
    elif imgy1>frame_height:
        imgy1=frame_height
        imgx1=(imgy1-b)/m
    
    if imgy2<0:
        imgy2 = 0
        imgx2 = -b/m
    elif imgy2>frame_height:
        imgy2=frame_height
        imgx2=(imgy2-b)/m
    return np.array([[frame_number,imgy1,imgx1],[frame_number,imgy2,imgx2]])