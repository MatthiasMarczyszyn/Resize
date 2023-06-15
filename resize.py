import cv2 
from scipy.interpolate import interp2d
import numpy as np 
import time
from PIL import Image


def read_and_resize(img_path,scale):
    img_source = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    shape = img_source.shape
    new_shape = (int(shape[0]/scale), int(shape[1]/scale)) 
    print(new_shape)
    img_resized = np.zeros(new_shape)
    for each_line in range(0, len(img_resized)):
        for each_pixel in range(0, len(img_resized[each_line])):
            img_resized[each_line][each_pixel] = img_source[each_line*scale -1][each_pixel*scale - 1]
           
    cv2.imwrite("resized.png", img_resized)
    return img_resized

def back_to_scale(img,scale):
    shape = (int(img.shape[0] * scale), int(img.shape[0] * scale))
    new_img = np.zeros(shape)
    for each_line in range(0, len(img)):
        for each_pixel in range(0, len(img[each_line])):
            new_img[each_line*scale -1][each_pixel*scale - 1] = img[each_line][each_pixel]
    
    cv2.imwrite("back.png", new_img)
    return new_img
    

def find_in_line(line, no_varaible, start_point):
    list_of_varaible = []
    
    for each_pixel in range(start_point, len(line)):
        if line[each_pixel] > 0:
            list_of_varaible.append(each_pixel)
        if len(list_of_varaible) == no_varaible:
            return list_of_varaible
    return None

def find_in_column(matrix, column , no_varaible, start_point):
    list_of_varaible = []
    
    for each_pixel in range(start_point, len(matrix)):
        if matrix[each_pixel][column] > 0:
            list_of_varaible.append(each_pixel)
        if len(list_of_varaible) == no_varaible:
            return list_of_varaible
    return None


def linear(img, scale):
    back_img = back_to_scale(img,scale)
    
    for each_line in range(1, len(back_img[0]) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_line(back_img[each_line], 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if back_img[each_line][pixel] == 0:
                    
                    back_img[each_line][pixel] = back_img[each_line][points[0]] + (pixel - points[0])*(back_img[each_line][points[0]] -back_img[each_line][points[1]])/(points[1]- points[0])
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(back_img) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_column(back_img, each_column, 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(back_img[each_column][points[0]])
                if back_img[pixel][each_column] == 0:
                    back_img[pixel][each_column] = back_img[points[0]][each_column] + (pixel - points[0])*(back_img[points[0]][each_column] -back_img[points[1]][each_column])/(points[1]- points[0])
            start = points[-1]
            
    
            
    cv2.imwrite("back_lin.png", back_img)
    print(compare(back_img))
    
def get_square_func(values, positions):
    
    a = (values[2] - values[1]) / ((positions[2] -positions[1]) * (positions[2] - positions[0])) - (values[1] - values[0]) / ((positions[1] - positions[0]) * (positions[2] - positions[0]))
    b = (values[2] - values[0]) / (positions[1] - positions[0]) - a * (positions[0] + positions[1])
    c = values[0] - a * positions[0]**2 - b*positions[0]
    return (c ,b, a)

def square(img, scale):
    back_img = back_to_scale(img,scale)
  
    
    for each_line in range(1, len(back_img[0]) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_line(back_img[each_line], 3, start)
            if points == None:
                break
            func_points = get_square_func([back_img[each_line][points[0]],back_img[each_line][points[1]],back_img[each_line][points[2]]],points)
            for pixel in range(start, points[-1]):
                if back_img[each_line][pixel] == 0:
                    back_img[each_line][pixel] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(back_img) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_column(back_img, each_column, 3, start)
            if points == None:
                break
            func_points = get_square_func([back_img[points[0]][each_column],back_img[points[1]][each_column],back_img[points[2]][each_column]],points)
            for pixel in range(start, points[-1]):
                if back_img[pixel][each_column] == 0:
                    back_img[pixel][each_column] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
            start = points[-1]
            
            
    cv2.imwrite("back_sqr.png", back_img)
    print(compare(back_img))

def get_cubic_func(values, positions, point):
    
    for i in range(1, len(positions) -1):
        if positions[i] < point and positions[i + 1] >point:
            h = positions[i+1] - positions[i]
            a = (positions[i+1] - point) / h
            b = (point - positions[i]) / h
            c = (a**3 - a) * (h**2)
            d = (b**3 - b) * (h**2)
            return a * values[i] + b * values[i+1] + c * (values[i] - values[i-1]) + d * (values[i+1] - values[i])

def cubic(img,scale):
    back_img = back_to_scale(img,scale)
    
    for each_line in range(1, len(back_img[0]) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_line(back_img[each_line], 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if back_img[each_line][pixel] == 0:
                    back_img[each_line][pixel] = get_cubic_func([back_img[each_line][points[0]-1],back_img[each_line][points[0]],back_img[each_line][points[1]],back_img[each_line][points[2]],back_img[each_line][points[3]]],[points[0] -1 ] + points,pixel)
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(back_img) -1):
        start = 1
        while start < len(back_img[0]) -1 :
            points =  find_in_column(back_img, each_column, 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(back_img[each_column][points[0]])
                if back_img[pixel][each_column] == 0:
                    back_img[pixel][each_column] = get_cubic_func([back_img[points[0]-1][each_column],back_img[points[0]][each_column],back_img[points[1]][each_column],back_img[points[2]][each_column],back_img[points[3]][each_column]],[points[0] -1 ]+ points,pixel)
            start = points[-1]
            
            
            
     
    cv2.imwrite("back_cub.png", back_img)
    print(compare(back_img))

def compare(img):
    orginal_img = cv2.imread("Grey.png",cv2.IMREAD_GRAYSCALE)
    
    return (np.square(np.subtract(orginal_img,img)).sum(), np.absolute(np.subtract(orginal_img,img)).sum())

if __name__ == "__main__":
    
    scale = 2
    resized = read_and_resize("Grey.png",scale)
    linear(resized,scale)
    square(resized,scale)
    cubic(resized,scale)