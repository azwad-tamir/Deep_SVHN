def convo(Window, im, axis):
    
    ########################
    #convolution function:
    ########################
    
    import numpy as np
    
    #calculating image dimensions
    length_x = im.shape[1]
    length_y = im.shape[0]
    #calculating window dimensions
    Wx = Window.shape[1]
    Wy = Window.shape[0]
    
    #flipping the window
    if axis==0:
        Window = np.flipud(Window)
    else:
        Window = np.fliplr(Window)
    
    #initializing output image
    D = np.zeros((length_y-Wy+1,length_x-Wx+1))

    #These for loops are used to go over every pixel and do the convolution operation
    for i in range(0,D.shape[0]):
        for j in range(0,D.shape[1]):
            temp = im[i:i+Wx,j:j+Wy]
            d = np.multiply(temp,Window)
            D[i,j] = np.sum(d)  #the summing function is used to sum over all the values of the filter
    
    return D

