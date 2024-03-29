import napari
from  napari_segment_everything import segment_everything

viewer = napari.Viewer()

viewer.window.add_dock_widget(segment_everything.NapariSegmentEverything(viewer))

k=input("press close to exit") 
