import cv2
import matplotlib.pyplot as plt

# Нахождение дублей, годящееся для нахождения при поиске в уже готовой большой БД
def intensities(image_path):
      '''
      https://gist.github.com/liamwhite/b023cdba4738e911293a8c610b98f987
      Алгоритм основан на анализе средней интенсивности изображения и его четвертей
      Данная функция возвращает вектор, соответствующий изображению, полученному на входе
      В принципе, можно получать и одномерные вектора, но меньшая размерность приводит к тому, что вектора чаще будут считаться похожими
      '''
      # Существует проблема с OpenCV, не позволяющая работать с файлами вне рабочей директории
      image = plt.imread(image_path)
      image = image[..., ::-1]
      #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
      image = cv2.GaussianBlur(image, (3,3), 0)

      half_x = round(image.shape[1] / 2)
      half_y = round(image.shape[0] / 2)

      nw_rect = image[0:half_y, 0:half_x]
      ne_rect = image[0:half_y, (half_x + 1):image.shape[1]]
      sw_rect = image[(half_y + 1):image.shape[0], 0:half_x]
      se_rect = image[(half_y + 1):image.shape[0], (half_x + 1):image.shape[1]]

      average_intensity = rect_sum(image)
      nw_intensity = rect_sum(nw_rect)
      ne_intensity = rect_sum(ne_rect)
      sw_intensity = rect_sum(sw_rect)
      se_intensity = rect_sum(se_rect)

      # Вектор, характеризующий изображение
      return average_intensity, nw_intensity, ne_intensity, sw_intensity, se_intensity

# Сумматор для получения координаты вектора
def rect_sum(rect):
    sums = cv2.sumElems(rect)
    r = (sums[0] / (rect.shape[0] * rect.shape[1])) * 0.2126
    g = (sums[1] / (rect.shape[0] * rect.shape[1])) * 0.7152
    b = (sums[2] / (rect.shape[0] * rect.shape[1])) * 0.0772

    return round((r + g + b) / 3, 6)

def intensities_iter(image_list):
   '''
   Итератор, возвращающий вектора для набора полученных изображений
   С помощью данной функции можно составить БД, по которой и будет производиться поиск изображений
   '''
   for image in image_list:
      yield intensities(image)

def similarity(vector1, vector2):
   '''
   Уровень похожести между векторами. 0 - идентичные, 1 - максимально разные
   '''
   distance = sum(map(lambda x1,x2:(x1-x2)**2,vector1,vector2))**.5
   norm1 = sum(x**2 for x in vector1)**.5
   norm2 = sum(x**2 for x in vector2)**.5
   
   return distance/(norm1+norm2)

def is_similar(vector1,vector2,threshold=0.1):
   return True if similarity(vector1,vector2)<threshold else False

def similarity_images(image1, image2):
   return similarity(intensities(image1), intensities(image2))

def is_similar_images(image1,image2,threshold=0.1):
   return True if similarity_images(image1, image2)<threshold else False