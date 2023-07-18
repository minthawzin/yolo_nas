import cv2

def addBboxToImage( x1, y1, x2, y2, image, color ):
    x1, y2, x2, y2 = map(int, [x1,y1,x2,y2])
    cv2.rectangle( image, ( x1, y1 ), ( x2, y2 ), color, thickness=2 )

    return image

def CXCY2XYXY( xc, yc, width, height ):
    x1 = xc - width/2
    x2 = xc + width/2
    y1 = yc - height/2
    y2 = yc + height/2

    return (int(x1), int(y1), int(x2), int(y2))

def preprocessCV( cv2_image, input_size ):
    original_image   = cv2.resize( cv2_image, (input_size, input_size))
    processed_img    = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB )   
    return processed_img        