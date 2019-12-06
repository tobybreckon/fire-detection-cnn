'''
padding images with 0 pixels to fit a particular size
'''



import numpy as np
def pad_images(output_size,image_np_array):
	new_height=output_size[0]
	new_width=output_size[1]
	[height,width,channels]=image_np_array.shape
	pad_height=np.floor((new_height-height)/2)
	pad_width=np.floor((new_width-width)/2)
	for j in range (0,channels):
		x=image_np_array[:,:,j]
		np.pad()


def resize_image(image,target_shape, pad_value = 0):
    assert isinstance(target_shape, list) or isinstance(target_shape, tuple)
    add_shape, subs_shape = [], []

    image_shape = image.shape
    shape_difference = np.asarray(target_shape, dtype=int) - np.asarray(image_shape,dtype=int)
    for diff in shape_difference:
        if diff < 0:
            subs_shape.append(np.s_[int(np.abs(np.ceil(diff/2))):int(np.floor(diff/2))])
            add_shape.append((0, 0))
        else:
            subs_shape.append(np.s_[:])
            add_shape.append((int(np.ceil(1.0*diff/2)),int(np.floor(1.0*diff/2))))
    output = np.pad(image, tuple(add_shape), 'constant', constant_values=(pad_value, pad_value))
    output = output[subs_shape]
    return output 
		

