# from numpy import *
# import matplotlib.pyplot as plt
#
#
# # 加载数据
# def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
#     dataMat = []  # 文件的最后一个字段是类别标签
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLines = line.strip().split()
#         fltLine = [float(curLine) for curLine in curLines]  # 将每个元素转成float类型
#         dataMat.append(fltLine)
#     return dataMat
#
#
# # 计算欧几里得距离
# def distEclud(vecA, vecB):
#     return sqrt(sum(power(vecA - vecB, 2)))  # 求两个向量之间的距离
#
#
# # 构建聚簇中心，取k个(此例中k=4)随机质心
# def randCent(dataSet, k):
#     n = shape(dataSet)[1]
#     centroids = mat(zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
#     for j in range(n):
#         minJ = min(dataSet[:, j])
#         maxJ = max(dataSet[:, j])
#         rangeJ = float(maxJ - minJ)
#         centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
#     return centroids
#
#
# # k-means 聚类算法
# def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
#     '''
#     :param dataSet:  没有lable的数据集  (本例中是二维数据)
#     :param k:  分为几个簇
#     :param distMeans:    计算距离的函数
#     :param createCent:   获取k个随机质心的函数
#     :return: centroids： 最终确定的 k个 质心
#             clusterAssment:  该样本属于哪类  及  到该类质心距离
#     '''
#     m = shape(dataSet)[0]  # m=80,样本数量
#     clusterAssment = mat(zeros((m, 2)))
#     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离，
#     centroids = createCent(dataSet, k)
#     clusterChanged = True  # 用来判断聚类是否已经收敛
#     while clusterChanged:
#         clusterChanged = False;
#         for i in range(m):  # 把每一个数据点划分到离它最近的中心点
#             minDist = inf;
#             minIndex = -1;
#             for j in range(k):
#                 distJI = distMeans(centroids[j, :], dataSet[i, :])
#                 if distJI < minDist:
#                     minDist = distJI;
#                     minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
#             if clusterAssment[i, 0] != minIndex:
#                 clusterChanged = True  # 如果分配发生变化，则需要继续迭代
#             clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典
#         # print centroids
#         for cent in range(k):  # 重新计算中心点
#             ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 去第一列等于cent的所有列
#             centroids[cent, :] = mean(ptsInClust, axis=0)  # 算出这些数据的中心点
#     return centroids, clusterAssment
#
#
# # --------------------测试----------------------------------------------------
# # 用测试数据及测试kmeans算法
# if __name__ == '__main__':
#     datMat = mat(loadDataSet(r'C:\Users\Administrator\Desktop\k-means-test.txt'))
#     myCentroids, clustAssing = kMeans(datMat, 3)
#     print(myCentroids)
#     print(clustAssing)
#
#     plt.figure(1)
#     x = array(datMat[:, 0])
#     y = array(datMat[:, 1])
#     plt.scatter(x, y, marker='o')
#     xcent = array(myCentroids[:, 0])
#     ycent = array(myCentroids[:, 1])
#     plt.scatter(xcent, ycent, marker='x', color='r', s=50)
#     plt.show()


# 其他实现方法
# from collections import defaultdict
# from random import uniform
# from math import sqrt
#
#
# def read_points():
#     dataset = []
#     with open(r'C:\Users\Administrator\Desktop\k-means-test.txt') as file:
#         for line in file:
#             if line == '\n':
#                 continue
#             dataset.append(list(map(float, line.strip().split())))
#         file.close()
#         return dataset
#
#
# def write_results(listResult, dataset, k):
#     with open('result.txt', 'a') as file:
#         for kind in range(k):
#             file.write("CLASSINFO:%d\n" % (kind + 1))
#             for j in listResult[kind]:
#                 file.write('%d\n' % j)
#             file.write('\n')
#         file.write('\n\n')
#         file.close()
#
#
# def point_avg(points):
#     dimensions = len(points[0])
#     new_center = []
#     for dimension in range(dimensions):
#         sum = 0
#         for p in points:
#             sum += p[dimension]
#         new_center.append(float("%.8f" % (sum / float(len(points)))))
#     return new_center
#
#
# def update_centers(data_set, assignments, k):
#     new_means = defaultdict(list)
#     centers = []
#     for assignment, point in zip(assignments, data_set):
#         new_means[assignment].append(point)
#     for i in range(k):
#         points = new_means[i]
#         centers.append(point_avg(points))
#     return centers
#
#
# def assign_points(data_points, centers):
#     assignments = []
#     for point in data_points:
#         shortest = float('inf')
#         shortest_index = 0
#         for i in range(len(centers)):
#             value = distance(point, centers[i])
#             if value < shortest:
#                 shortest = value
#                 shortest_index = i
#         assignments.append(shortest_index)
#     if len(set(assignments)) < len(centers):
#         print("\n--!!!产生随机数错误，请重新运行程序！!!!--\n")
#         exit()
#     return assignments
#
#
# def distance(a, b):
#     dimention = len(a)
#     sum = 0
#     for i in range(dimention):
#         sq = (a[i] - b[i]) ** 2
#         sum += sq
#     return sqrt(sum)
#
#
# def generate_k(data_set, k):
#     centers = []
#     dimentions = len(data_set[0])
#     min_max = defaultdict(int)
#     for point in data_set:
#         for i in range(dimentions):
#             value = point[i]
#             min_key = 'min_%d' % i
#             max_key = 'max_%d' % i
#             if min_key not in min_max or value < min_max[min_key]:
#                 min_max[min_key] = value
#             if max_key not in min_max or value > min_max[max_key]:
#                 min_max[max_key] = value
#     for j in range(k):
#         rand_point = []
#         for i in range(dimentions):
#             min_val = min_max['min_%d' % i]
#             max_val = min_max['max_%d' % i]
#             tmp = float("%.8f" % (uniform(min_val, max_val)))
#             rand_point.append(tmp)
#         centers.append(rand_point)
#     return centers
#
#
# def k_means(dataset, k):
#     k_points = generate_k(dataset, k)
#     assignments = assign_points(dataset, k_points)
#     old_assignments = None
#     while assignments != old_assignments:
#         new_centers = update_centers(dataset, assignments, k)
#         old_assignments = assignments
#         assignments = assign_points(dataset, new_centers)
#     result = list(zip(assignments, dataset))
#     print('\n\n---------------------------------分类结果---------------------------------------\n\n')
#     for out in result:
#         print(out, end='\n')
#     print('\n\n---------------------------------标号简记---------------------------------------\n\n')
#     listResult = [[] for i in range(k)]
#     count = 0
#     for i in assignments:
#         listResult[i].append(count)
#         count = count + 1
#     write_results(listResult, dataset, k)
#     for kind in range(k):
#         print("第%d类数据有:" % (kind + 1))
#         count = 0
#         for j in listResult[kind]:
#             print(j, end=' ')
#             count = count + 1
#             if count % 25 == 0:
#                 print('\n')
#         print('\n')
#     print('\n\n--------------------------------------------------------------------------------\n\n')
#
#
# def main():
#     dataset = read_points()
#     k_means(dataset, 3)
# if __name__ == "__main__":
#   main()


import xml.etree.ElementTree as ET
import numpy as np
import cfg


def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的交并比的均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数,求中位数
    返回值：形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离,初始化时赋值很小的数
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 设置随机数种子
    np.random.seed()

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters

    return clusters


# 加载自己的数据集，只需要所有labelimg标注出来的xml文件即可
def load_dataset(path):
    # dataset = []
    # for xml_file in glob.glob("{}/*xml".format(path)):
    #     tree = ET.parse(xml_file)
    #     # 图片高度
    #     height = int(tree.findtext("./size/height"))
    #     # 图片宽度
    #     width = int(tree.findtext("./size/width"))
    #
    #     for obj in tree.iter("object"):
    #         # 偏移量
    #         xmin = int(obj.findtext("bndbox/xmin")) / width
    #         ymin = int(obj.findtext("bndbox/ymin")) / height
    #         xmax = int(obj.findtext("bndbox/xmax")) / width
    #         ymax = int(obj.findtext("bndbox/ymax")) / height
    #         xmin = np.float64(xmin)
    #         ymin = np.float64(ymin)
    #         xmax = np.float64(xmax)
    #         ymax = np.float64(ymax)
    #         if xmax == xmin or ymax == ymin:
    #             print(xml_file)
    #         # 将Anchor的长宽放入dateset，运行kmeans获得Anchor
    #         dataset.append([xmax - xmin, ymax - ymin])
    # return np.array(dataset)

    boxes = []
    original_size = np.array([[292., 791.],
                              [516., 375.],
                              [431., 1206.],
                              [755., 1206.],
                              [378., 521.],
                              [636., 512.],
                              [1263., 3013.],
                              [2026., 3145.],
                              [268., 405.],
                              [297., 426.],
                              [483., 717.],
                              [3276., 2294.],
                              [524., 603.],
                              [490., 521.],
                              [644., 488.],
                              [658., 566.],
                              [521., 613.],
                              [733., 2274.]])
    # print(original_size)
    with open(path) as text:
        dataset = text.readlines()
    for str in dataset:
        strs = str.strip().split()
        box = np.array([float(x) for x in strs[1:]])
        box = np.split(box, len(box) // 5)
        boxes.extend(box)
    boxes = np.stack(boxes)[:, 2:4] / original_size
    return boxes


if __name__ == '__main__':
    ANNOTATIONS_PATH = cfg.TEXT_PATH  # xml文件所在文件夹
    CLUSTERS = 9  # 聚类数量，anchor数量
    INPUTDIM = 416  # 输入网络大小

    # 求得真实框相对原图大小的比例，后续Kmeans出来的结果就是比值
    data = load_dataset(ANNOTATIONS_PATH)
    print(data)
    out = kmeans(data, k=CLUSTERS)
    print('Boxes:')
    print(np.array(out) * INPUTDIM)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))
