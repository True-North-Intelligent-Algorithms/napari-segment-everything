# import skimag example images
from skimage import data
from napari_segment_everything.sam_helper import get_mobileSAMv2, get_bounding_boxes

def test_mobile_sam():
    # load a color examp
    image = data.coffee()
    
    bounding_boxes = get_bounding_boxes(image, imgsz=1024, device = 'cuda')
    segmentations = get_mobileSAMv2(image, bounding_boxes)
    
    assert len(segmentations) == 11