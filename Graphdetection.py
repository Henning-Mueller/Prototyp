import math
import os
import cv2
import numpy as np
import tensorflow as tf
import random
import Gates
import time
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from termcolor import colored

def start(soloparam = "f4746511-2473-46b0-814f-395498b94de4"):
    detection_model, category_index = loadModel()
    #startsolo(detection_model, category_index, soloparam)
    startall(detection_model, category_index)
def startall(detection_model, category_index):

    for i in os.listdir("images"):
        start_time = time.time()
        startsolo(detection_model, category_index,str(i))
        end_time = time.time()
        spent_time = round(end_time - start_time,2)

        print("Es wurden " + str(spent_time) +" Sekunden für die Verarbeitung des Bildes benötigt")
        print("\n")



def startsolo(detection_model, category_index,IMAGE_name):
    global IMAGE_NAME
    global IMAGE_PATH
    global MINSCORE
    global def_dir_offset
    MINSCORE = 0.75
    def_dir_offset = 6
    IMAGE_NAME = IMAGE_name
    IMAGE_PATH = os.path.join("images", IMAGE_NAME)
    print("Image: " + colored(IMAGE_NAME,"blue") + " with Min-Detections-Score of: " + str(MINSCORE))
    image_np = load(IMAGE_NAME, IMAGE_PATH)
    detections = run_inference(detection_model, image_np)
    IwD = detectFromImage(category_index, image_np, detections)
    boxes, imageblack = drawboxes(detections, image_np, category_index)
    pre = preprocessing(imageblack)
    skel = skelonization(pre)
    classified_pixels, port_pixels = getclassifiedpixels(skel)
    uninterrupted_Edges, crossings, = identify_devide_edge_sections(
        classified_pixels, port_pixels)
    crossings, boundingboxes = merge_crossingpixels(crossings,classified_pixels)
    merged_sections = Connect_sections(classified_pixels, crossings, port_pixels,
                                       boundingboxes)
    edge_sections = uninterrupted_Edges + merged_sections
    connections = connectNodes(edge_sections, boxes)
    dc = drawconections(edge_sections,image_np)
    elements = build(boxes, connections)
    solve(elements, boxes, image_np)
    TabelleColored(elements)


"""
Läd das Trainierte ObjectDetection Model
Output: Labelset und Model
"""
def loadModel():
    utils_ops.tf = tf.compat.v1
    tf.gfile = tf.io.gfile
    PATH_TO_LABELS = 'SSDMobileNet/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    detection_model = tf.saved_model.load('SSDMobileNet/saved_model')
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore('SSDMobileNet/checkpoint/ckpt-0').expect_partial()
    return detection_model,category_index


"""
Läd das Image aus dem Imagepath
Output:
"""
def load(IMAGE_NAME, IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH + "/" + IMAGE_NAME + ".jpg")
    image_np = np.array(img)
    return image_np

"""
Führt die Objecterkennung auf dem Bild aus
Output: Das Detection Array mit allen Informationen über die Detections
"""
def run_inference(model, image_np):
    image_np = np.asarray(image_np)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                 for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict

"""
Erstellt ein Bild zu den detections
Zeichnet dazu die Boundingboxen in das Bild
"""
def detectFromImage(category_index, image_np, detections):
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=15,
        min_score_thresh=MINSCORE,
        agnostic_mode=False)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_detections' + '.png'), image_np_with_detections)
    return image_np_with_detections
"""
Verarbeitund der Erkannten Elemente
1.Auswahl der Detections für die Verarbeitung anhand des detections Scores
2.Erstellung des Boxes Arrays(Posdaten + DectKlasse)
3.Erstellung eines Bildes welches die Elemente im Bild Nummeriert
4.Erstellung des Bildes in welchem die Detection Bereiche entfernt wurden
    Dazu wird 1 Evaluationsbild erstellt welches die entfernten Bereiche darstellt
5. Ausgabe Array aus Schritt 2 und Bild aus Schritt 4
"""
def drawboxes(detections, image_np,catgory_index):
    count = 0
    for i in detections['detection_scores']:
        if i >= MINSCORE:
            count += 1
    boxes = np.zeros((count, 5))
    box_pixel = np.zeros((image_np.shape[0], image_np.shape[1], image_np.shape[2]))
    imageblack = image_np.copy()
    image_np2 = image_np.copy()
    for q in range(0, count):
        xmin = detections['detection_boxes'][q][0] * image_np.shape[0] - 3
        ymin = detections['detection_boxes'][q][1] * image_np.shape[1] - 3
        xmax = detections['detection_boxes'][q][2] * image_np.shape[0] + 3
        ymax = detections['detection_boxes'][q][3] * image_np.shape[1] + 3
        cv2.putText(image_np2, str(q) + ": " + str(catgory_index[detections['detection_classes'][q]]["name"]),
                    (int((ymin + ymax) / 2 -55), int((xmin + xmax) / 2 - 40)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)
        boxes[q] = [int(xmin), int(ymin), int(xmax), int(ymax), detections['detection_classes'][q]]
        imageblack[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
        box_pixel[int(xmin):int(xmax), int(ymin):int(ymax)] = (255, 255, 255)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_numbers' + '.png'), image_np2)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_imageblack' + '.png'), imageblack)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_box_pixel' + '.png'), box_pixel)
    return boxes, imageblack

"""
Begin Graph Detection
Preprocessing
1. Color -> Gray Convertion des Bildes
2. Otsu Thresholding
3. Umwandeln in Bin Bild also Demensions Reduktion, Invertierung und von 0/255 auf 0/1 redzuiert
4. Output PreProcesed Image (ppimg)
"""
def preprocessing(imageblack):
    imgray = cv2.cvtColor(imageblack, cv2.COLOR_BGR2GRAY)
    otsu_threshold, ppimg = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ppimg = 255-ppimg
    ppimg = ppimg /255
    ppimg = ppimg.astype(np.float64)
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_preprocess' + '.png'), ppimg*255)
    return ppimg


"""
1. Thining des bildes via Zhang and Suen
2. Morphologischer Überprüfungsalgorithmus
3. Erstellung Evaluations Bild Nach Thining und Morphologischer bereinigung
"""
def skelonization(pre):
    BS = np.copy(pre)
    BS = zhang_and_suen_binary_thinning(BS)
    for x in range(1, BS.shape[0] - 1):
        for y in range(1, BS.shape[1] - 1):
            if BS[x, y] == 1:
                d_1 = (x > 2) and (BS[x - 2, y - 1] == 0 or BS[x - 2, y] == 1 or BS[x - 1, y - 1] == 1)
                d_2 = (y > 2) and (BS[x + 1, y - 2] == 0 or BS[x, y - 2] == 1 or BS[x + 1, y - 1] == 1)
                d_3 = (x < BS.shape[0] - 2) and (
                        BS[x + 2, y + 1] == 0 or BS[x + 2, y] == 1 or BS[x + 1, y + 1] == 1)
                d_4 = (y < BS.shape[1] - 2) and (
                        BS[x - 1, y + 2] == 0 or BS[x, y + 2] == 1 or BS[x - 1, y + 1] == 1)
                if BS[x - 1, y + 1] == 1 and (BS[x - 1, y] == 1 and d_1):
                    BS[x - 1, y] = 0
                if BS[x - 1, y - 1] == 1 and (BS[x, y - 1] == 1 and d_2):
                    BS[x, y - 1] = 0
                if BS[x + 1, y - 1] == 1 and (BS[x + 1, y] == 1 and d_3):
                    BS[x + 1, y] = 0
                if BS[x + 1, y + 1] == 1 and (BS[x, y + 1] == 1 and d_4):
                    BS[x, y + 1] = 0
    skel = BS
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_skel' + '.png'), skel*255)
    return skel

"""
Classifizierung der einzelnen Pixel im Bild
    in Edge-Pixel, Portpixel, Crossingpixel
    und Sammelung der Portpixel positionen
"""
def getclassifiedpixels(skel):
    classified_pixels = np.zeros((skel.shape[0], skel.shape[1]))
    port_pixels = []
    for x in range(1, skel.shape[0] - 1):
        for y in range(1, skel.shape[1] - 1):
            if skel[x, y] == 1:
                # 8-Neighborhood von Pixel (x, y)
                neighborhood = np.array(
                    [skel[x - 1, y], skel[x + 1, y],
                     skel[x, y - 1], skel[x, y + 1],
                     skel[x + 1, y + 1], skel[x + 1, y - 1],
                     skel[x - 1, y + 1], skel[x - 1, y - 1]])
                number_of_neighbours = np.sum(neighborhood)
                if number_of_neighbours == 2:
                    classified_pixels[x, y] = 2  # edge pixels
                elif number_of_neighbours > 2:
                    classified_pixels[x, y] = 3  # crossing pixels
                elif number_of_neighbours < 2:
                    classified_pixels[x, y] = 4  # port pixels
                    port_pixels.append((x, y))
    return classified_pixels, port_pixels


"""
Unterteilung und identifizierung der Edges
Abtastung der einzelnen Edges von Startpixel bis entweder Crossinpixel oder Portpixel
In falle von CrossingPixel einteiln in CPArray
In falle von Portpixel in Abgeschlossene Linien einteilen
"""
def identify_devide_edge_sections(classified_pixels, port_pixels):

    uninterrupted_Edges = []
    start_pixels = dict.fromkeys(port_pixels, 0)
    crossings = {}
    for start in start_pixels:
        # if port pixel is already visited, then continue
        if start_pixels[start] == 1:
            continue
        else:
            start_pixels[start] = 1
            section = []
            section.append(start)
            x, y = start
            neighbor = None
            neighbor_value = -1

            for p in range(0, 3):
                for q in range(0, 3):
                    if (p != 1 or q != 1) and (classified_pixels[x + p - 1, y + q - 1] > neighbor_value):
                        neighbor = np.array([x + p - 1, y + q - 1])
                        neighbor_value = classified_pixels[x + p - 1, y + q - 1]

            next = neighbor
            next_value = neighbor_value
            direction = np.subtract(next, start)

            while next_value == 2:  # edge pixel
                section.append(next)
                neighbor = None
                neighbor_value = -1
                x, y = next
                for p in range(0, 3):
                    for q in range(0, 3):
                        if (p != 1 or q != 1) and (p != 1 - direction[0] or q != 1 - direction[1]) and (
                                classified_pixels[x + p - 1, y + q - 1] > neighbor_value):
                            neighbor = np.array([x + p - 1, y + q - 1])
                            neighbor_value = classified_pixels[x + p - 1, y + q - 1]
                next_value = neighbor_value
                direction = np.subtract(neighbor, next)
                next = neighbor

            section.append(next)
            last_element = next
            next = last_element
        next_value = classified_pixels[next[0], next[1]]
        if next_value == 4:  # port pixel
            # marks the next pixel as already visited
            start_pixels[(next[0], next[1])] = 1
            uninterrupted_Edges.append(section)
        elif next_value == 3:  # crossing pixel
            pos = (next[0], next[1])
            if not pos in crossings:
                crossings[pos] = []
            crossings[pos].append(section)
    start_pixels.clear()
    return uninterrupted_Edges, crossings



"""
Verschmilzt alle nahe bei einander liegenden CPs zu einer Kreuzung
Bereitet Boundingboxen für die Kreuzungsverarbeitung vor
"""
def merge_crossingpixels(crossings, classified_pixels):
    all_crossing_pixels = np.stack(np.where(classified_pixels == 3), axis=1)
    for allc in all_crossing_pixels:
        if tuple(allc) not in crossings.keys():
            crossings[tuple(allc)] = []
    clusters = [[c] for c in crossings.items()]
    merge_happened = True
    new_crossings = {}
    boundingboxes = {}
    while merge_happened:
        merge_happened = try_to_merge(clusters)
    for c in clusters:
        center = tuple(np.round(np.array([p[0] for p in c]).mean(axis=0)).astype(int))
        boundingrect = cv2.boundingRect(np.asarray([np.asarray(x[0]) for x in c]))
        for x in range(boundingrect[0], boundingrect[0] + boundingrect[2]):
            for y in range(boundingrect[1], boundingrect[1] + boundingrect[3]):
                if classified_pixels[x, y] == 2:
                    classified_pixels[x, y] = 3
        new_sections = []
        for cp in c:
            for cpl in cp[1]:
                if not np.array_equal(cpl[-1], np.array(center)):
                    cpl.append(np.array(center))
            new_sections += cp[1]
        new_crossings[center] = new_sections
        boundingboxes[center] = boundingrect
    return new_crossings, boundingboxes


"""

"""
def try_to_merge(clusters):
    thresh = 8.6 # Smaler Circle = 7.1 bigger = 8.5
    for cluster in clusters:
        for crossing_pixel in cluster:
            for other_cluster in clusters:
                if (other_cluster == cluster):
                    continue
                for other_crossing_pixel in other_cluster:
                    if (euclid_tuple(crossing_pixel[0], other_crossing_pixel[0]) < thresh):
                        cluster += other_cluster
                        clusters.remove(other_cluster)
                        return True
    return False



"""

"""
def euclid_tuple(t1, t2):
    return np.linalg.norm(np.array(tuple(map(lambda i, j: i - j, t1, t2))))



"""
Findet die nochnicht identifizierten Liniensectionen
Diese lingen zwischen 2 Crossingpixels.
Um linien inherlab einer Kreuzung nicht zu enddecken werden die Boundingboxen   verwendet
"""
def find_missing_sections(crossings, classified_pixels, sections, crossing_pixel, boundingboxes):
    found = True
    while found:
        found = find_missing(crossing_pixel, sections, classified_pixels, boundingboxes[crossing_pixel])
        if not found is None:
            crossings[crossing_pixel].append(found)
            found = True
        else:
            found = False
    return  crossings


def find_missing(crossing_pixel, sections, classified_pixels, boundingbox):
    for s in sections:
        s[0] = np.array(s[0])
    #radiust = 3  # smaler circle = 3
    for i in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
        for j in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
            currentpos = ( i, j)
            if classified_pixels[currentpos] == 3:
                back = currentpos
                radius = 1
                for p in range(-radius, +radius + 1):
                    for q in range(-radius, +radius + 1):
                        if p == 0 and q == 0:
                            continue
                        nextpos = (currentpos[0] + p, currentpos[1] + q)
                        if classified_pixels[nextpos] == 2 and not any(
                                [np.any(np.all(np.array(nextpos) == s, axis=1)) for s in sections]):
                            next = nextpos
                            section = [np.array(crossing_pixel)]
                            section += get_basic_section(next, classified_pixels, back)
                            boundingboxpx = []
                            for i in range(boundingbox[0], boundingbox[0] + boundingbox[2]):
                                for j in range(boundingbox[1], boundingbox[1] + boundingbox[3]):
                                    boundingboxpx.append(np.array([i,j]))
                            if np.any([np.all(section[-1] == bp) for bp in boundingboxpx]) and np.any([np.all(section[0] == bp) for bp in boundingboxpx]):
                                continue

                            return section

    return None



"""
Ähnlich aufgebaut wie identify and devide
"""
def get_basic_section(start, classified_pixels, back):
    delta = np.array([0, 0])
    section = []
    section.append(np.array(back))
    section.append(np.array(start))
    x, y = start
    neighbor = None
    neighbor_value = -float('inf')

    for i in range(0, 3):
        for j in range(0, 3):
            if (i != 1 or j != 1) and (x + i - 1 != back[0] or y + j - 1 != back[1]) and (
                    classified_pixels[x + i - 1, y + j - 1] == 2):
                neighbor = np.array([x + i - 1, y + j - 1])
                neighbor_value = classified_pixels[x + i - 1, y + j - 1]

    next = neighbor
    next_value = neighbor_value
    delta = np.subtract(next, start)

    while next_value == 2:  # edge pixel
        section.append(next)
        neighbor = None
        neighbor_value = -float('inf')
        x, y = next
        for i in range(0, 3):
            for j in range(0, 3):
                if (i != 1 or j != 1) and (i != 1 - delta[0] or j != 1 - delta[1]) and (
                        classified_pixels[x + i - 1, y + j - 1] > neighbor_value):
                    neighbor = np.array([x + i - 1, y + j - 1])
                    neighbor_value = classified_pixels[x + i - 1, y + j - 1]
        next_value = neighbor_value
        delta = np.subtract(neighbor, next)
        next = neighbor
    if next_value == 3:
        section.append(next)
    return section



"""
Gibt den Eintrittsvektor der einzelen Lininenabschitte welche an einer Kreuzung anliegen zurück
"""
def get_section_dir(sections,crossing_pixel):
    sectiondirs = []
    for cur_section in sections:
        if len(cur_section) < def_dir_offset:
            if (len(cur_section) <= 1):
                print("Linienabschnitslänge ist zu klein")
            dir_offset = len(cur_section)
        else:
            dir_offset = def_dir_offset  # if crossing not allways at the end set to +/-
        if np.array_equal(cur_section[0], np.array(crossing_pixel)):
            start_pixel = cur_section[1]
            end_pixel = cur_section[dir_offset]
        else:
            start_pixel = cur_section[-2]
            end_pixel = cur_section[-dir_offset]
        section_dir = end_pixel - start_pixel
        sectiondirs.append([cur_section, section_dir])
    return sectiondirs


"""
Verbindet Linien welche unterbrochen sind miteinander
"""
def Connect_sections(classified_pixels, crossings, port_pixels, boundingboxes):
    merged_sections = []
    # classified_pixelsf2 = copy.deepcopy(classified_pixels)
    for (crossing_pixel, sections) in crossings.items():
        crossings = find_missing_sections(crossings,classified_pixels, sections, crossing_pixel,
                                          boundingboxes)
        sectiondirs = get_section_dir(sections, crossing_pixel)
        merged_sections = merge_crossings(sectiondirs, crossing_pixel, merged_sections)
    return merge_overlapping_sections(port_pixels, merged_sections, classified_pixels)



"""
Ordnet die an eine Kreuzung anliegenden Edge_sections einander Zu und verschmiltzt zusammengehörende
"""
def merge_crossings(sectiondirs, crossing_pixel, merged_sections):
    skip_sections = []
    for first_section in sectiondirs:
        angle = []
        if section_in_sections(first_section[0], skip_sections):
            continue
        for other_section in sectiondirs:
            if section_equal(first_section[0], other_section[0]):
                continue
            angle.append([calcangle(first_section[1], other_section[1]), other_section[0]])
        angle.sort(key=lambda x: x[0], reverse=True)
        skip_sections.append(angle[0][1])
        if np.array_equal(first_section[0][-1], np.array(crossing_pixel)):
            if np.array_equal(angle[0][1][-1], np.array(crossing_pixel)):
                first_section[0].reverse()
                merged_sections.append(angle[0][1] + first_section[0][1:])
            elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                merged_sections.append(first_section[0][:-1] + angle[0][1])
            else:
                print("ERORR Traversal subphase 1")
        elif np.array_equal(first_section[0][0], np.array(crossing_pixel)):
            if np.array_equal(angle[0][1][-1], np.array(crossing_pixel)):
                merged_sections.append(angle[0][1] + first_section[0][1:])
            elif np.array_equal(angle[0][1][0], np.array(crossing_pixel)):
                first_section[0].reverse()
                merged_sections.append(first_section[0][:-1] + angle[0][1])
            else:
                print("ERORR Traversal subphase 2")
        else:
            print("ERORR Traversal subphase 0")
    return merged_sections


"""
Verschmilzt Überschneidene Linienabschnitte
"""
def merge_overlapping_sections(port_pixels, merged_sections,classified_pixels):
    merged = True
    while merged:
        merged = False
        for s1 in merged_sections:
            for s2 in merged_sections:
                if section_equal(s1, s2):
                    continue
                if not ((s1[0][0], s1[0][1]) in port_pixels and (s1[-1][0], s1[-1][1]) in port_pixels):
                    if not ((s2[0][0], s2[0][1]) in port_pixels and (s2[-1][0], s2[-1][1]) in port_pixels):
                        if any(np.all(s1[-1] == s2, axis=1)) and any(np.all(s1[-2] == s2, axis=1)):
                            w1 = np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]
                            w2 = np.where(np.all(s1[-2] == s2, axis=1) == True)[0][0]
                            if w1 > w2:
                                s3 = s2[np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]:]
                                s4 = s1 + s3
                            else:
                                s3 = s2[:np.where(np.all(s1[-1] == s2, axis=1) == True)[0][0]]
                                s3.reverse()
                                s4 = s1 + s3
                            remove_section_from_sections(s1, merged_sections)
                            remove_section_from_sections(s2, merged_sections)
                            merged_sections.append(s4)
                            merged = True
                            break
                        elif any(np.all(s1[0] == s2, axis=1)):
                            # skip da s1 und s2 vertauscht sind wird später richtigherum aufgerufen
                            pass
                        else:
                            pass
                            # print("H3")
    return merged_sections


"""
Berechnet den Winkel zwischen zwei Vektoren(Linien Eintritswinkel)
"""
def calcangle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(max(-1,min(dot_product,1)))
    return np.degrees(angle)


def section_equal(a, b):
    return np.array_equal(np.array(a), np.array(b))


def section_in_sections(section, sections):
    for s in sections:
        if section_equal(s, section):
            return True
    return False

def remove_section_from_sections(section, sections):
    counter = 0
    for s in sections:
        if section_equal(s, section):
            del sections[counter]
        counter+=1

"""
Wandelt Edge_sections in conections um
Conections geben an welches element mit welchem verbundne ist
"""
def connectNodes(edge_sections, boxes):
    connections = {}
    counter = 0
    for section in edge_sections:
        if section[0][1] < section[-1][1]:
            connections[counter] = [[section[0],section[-1]], [-1, -1]]
            c = 0
            for box in boxes:
                if (box[3] == section[0][1] or box[3] + 1 == section[0][1]) and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][0] = c
                if (box[1] == section[-1][1] + 1 or box[1] == section[-1][1] + 2) and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][1] = c
                c += 1
        else:
            connections[counter] = [[section[-1], section[0]], [-1, -1]]
            c = 0
            for box in boxes:
                if (box[3] == section[-1][1] or box[3] + 1 == section[-1][1]) and box[0] < section[-1][0] and box[2] > section[-1][0]:
                    connections[counter][1][0] = c
                if (box[1] == section[0][1] + 1 or box[1] == section[0][1] + 2) and box[0] < section[0][0] and box[2] > section[0][0]:
                    connections[counter][1][1] = c

                c += 1
        counter += 1
    return connections


"""
Gibt eine Wahrheitstabelle zu der Analysierten Schaltung in der Konsole aus
In dieser MEthode sind Ture und False werte Fablich makiert
"""
def TabelleColored(elements):
    inputelements = []
    outputelements = []

    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].setFalse()
            inputelements.append(element)
        elif type(element[0]) == Gates.Output:
            outputelements.append(element)
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\tIn: " + str(element[1]), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\tOut:" + str(element[1]), end="\t|")
    print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")
    print("|", end="")
    for element in inputelements:
        print("\t", end="")
        if element[0].getValue() == True:
            print(colored(str(element[0].getValue()), "green"), end="\t|")
        else:
            print(colored(str(element[0].getValue()), "red"), end="\t|")
    print("\t|", end="")
    for element in outputelements:
        print("\t", end="")
        if element[0].getValue() == True:
            print(colored(str(element[0].getValue()), "green"), end="\t|")
        else:
            print(colored(str(element[0].getValue()), "red"), end="\t|")
    print("")
    for _ in range(1, int(math.pow(2, len(inputelements)))):
        for _ in range(0, len(inputelements)):
            print("+ - - - - - ", end="")
        print(end="+ - ")
        for _ in range(0, len(outputelements)):
            print("+ - - - - - ", end="")
        print("+")
        scounter = 0
        switched = True
        while switched and scounter < len(inputelements):
            switched = not inputelements[scounter][0].switch()
            scounter += 1
        print("|", end="")
        for element in inputelements:
            print("\t", end="")
            if element[0].getValue() == True:
                print(colored(str(element[0].getValue()), "green"), end="\t|")
            else:
                print(colored(str(element[0].getValue()), "red"), end="\t|")

        print("\t|", end="")
        for element in outputelements:
            print("\t", end="")
            if element[0].getValue() == True:
                print(colored(str(element[0].getValue()), "green"), end="\t|")
            else:
                print(colored(str(element[0].getValue()), "red"), end="\t|")
        print("")
    for _ in range(0, len(inputelements)):
        print("+ - - - - - ", end="")
    print(end="+ - ")
    for _ in range(0, len(outputelements)):
        print("+ - - - - - ", end="")
    print("+")


"""
Färbt die Erkannten Linien im Bild mit einer Unique Color ein
Evaluations Bild
"""
def drawconections(edge_sections,image_np):
    Colors =[]
    img_con = image_np.copy()
    r = lambda: max(random.randint(0, 4) * 64 - 1, 0)
    while len(Colors) < len(edge_sections):
        curColor =(r(), r(), r())
        if sum(list(curColor)) > 637:
            continue
        if curColor in Colors:
            continue
        Colors.append(curColor)
    counterc = 0
    for edge_section in edge_sections:
        for px in edge_section:
            img_con[px[0], px[1]] = Colors[counterc]
            img_con[px[0], px[1] + 1] = Colors[counterc]
            img_con[px[0] + 1, px[1]] = Colors[counterc]
            img_con[px[0], px[1] - 1] = Colors[counterc]
            img_con[px[0] - 1, px[1]] = Colors[counterc]
        counterc+=1
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_connections' + '.png'), img_con)
    return img_con

"""
Erstellt die Lösung zu den im Bild definierten Inputs.
Gibt diese in der Console und In Bildform aus
"""
def solve(elements, boxes, image_np):
    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].update()
    print("Lösung")
    count = 0
    for element in elements:
        if type(element[0]) == Gates.Output:
            print("Output " + str(element[1]) + ": " + str(element[0].getValue()))
            cv2.putText(image_np,str(element[0].getValue()),
                (int((boxes[count][1] + boxes[count][3]) / 2 -20), int((boxes[count][0] + boxes[count][2]) / 2 -40)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        count += 1
    cv2.imwrite(os.path.join('images' + "/" + IMAGE_NAME + "/", IMAGE_NAME + '_Solution' + '.png'), image_np)


"""
Konstruiert anhand der Extrahierten daten die Schaltung
"""
def build(boxes, connections):
    elements = []
    alt = 0
    label_offset = 1
    for box in boxes:
        if box[4] == 0 + label_offset:
            # print("switch_True")
            elements.append([Gates.Input(True), alt])
        elif box[4] == 1 + label_offset:
            # print("switch_False")
            elements.append([Gates.Input(), alt])
        elif box[4] == 2 + label_offset:
            # print("and")
            elements.append([Gates.And(), alt])
        elif box[4] == 3 + label_offset:
            # print("or")
            elements.append([Gates.Or(), alt])
        elif box[4] == 4 + label_offset:
            # print("nand")
            elements.append([Gates.Nand(), alt])
            # elements[-1].addInput(alt)
        elif box[4] == 5 + label_offset:
            # print("nor")
            elements.append([Gates.Nor(), alt])
        elif box[4] == 6 + label_offset:
            # print("not")
            elements.append([Gates.Not(), alt])
        elif box[4] == 7 + label_offset:
            # print("bulb")
            elements.append([Gates.Output(), alt])
        alt += 1
    for connection in connections.values():
        print("Connect: " + str(connection[1][0]) + " mit " + str(connection[1][1]))
        if not type(elements[connection[1][1]][0]) == Gates.Input:
            elements[connection[1][1]][0].addInput(elements[connection[1][0]][0])
    return elements


"""
Thinningalrotimus nach Zhang und Suen
Implementation von  Alexsander Andrade de Melo
"""
def zhang_and_suen_binary_thinning(A):

    height = A.shape[0]
    width = A.shape[1]

    _A = np.copy(A)

    removed_points = []
    flag_removed_point = True

    while flag_removed_point:
        flag_removed_point = False
        for x in range(1, height - 1):
            for y in range(1, width - 1):
                if _A[x, y] == 1:
                    # get 8-neighbors
                    neighborhood = [_A[x - 1, y], _A[x - 1, y + 1], _A[x, y + 1], _A[x + 1, y + 1],
                                    _A[x + 1, y], _A[x + 1, y - 1], _A[x, y - 1], _A[x - 1, y - 1]]
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

                    # B_P1 is the number of nonzero neighbors of P1=(x, y)
                    B_P1 = np.sum(neighborhood)
                    condition_1 = 2 <= B_P1 <= 6
                    # A_P1 is the number of 01 patterns in the ordered set of neighbors
                    n = neighborhood + neighborhood[0:1]
                    A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

                    condition_2 = A_P1 == 1
                    condition_3 = P2 * P4 * P6 == 0
                    condition_4 = P4 * P6 * P8 == 0

                    if (condition_1 and condition_2 and condition_3 and condition_4):
                        removed_points.append((x, y))
                        flag_removed_point = True

        for x, y in removed_points:
            _A[x, y] = 0
        del removed_points[:]

        for x in range(1, height - 1):
            for y in range(1, width - 1):

                if _A[x, y] == 1:
                    # get 8-neighbors
                    neighborhood = [_A[x - 1, y], _A[x - 1, y + 1], _A[x, y + 1], _A[x + 1, y + 1],
                                    _A[x + 1, y], _A[x + 1, y - 1], _A[x, y - 1], _A[x - 1, y - 1]]
                    P2, P3, P4, P5, P6, P7, P8, P9 = neighborhood

                    # B_P1 is the number of nonzero neighbors of P1=(x, y)
                    B_P1 = np.sum(neighborhood)
                    condition_1 = 2 <= B_P1 <= 6

                    # A_P1 is the number of 01 patterns in the ordered set of neighbors
                    n = neighborhood + neighborhood[0:1]
                    A_P1 = sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

                    condition_2 = A_P1 == 1
                    condition_3 = P2 * P4 * P8 == 0
                    condition_4 = P2 * P6 * P8 == 0

                    if (condition_1 and condition_2 and condition_3 and condition_4):
                        removed_points.append((x, y))
                        flag_removed_point = True

        for x, y in removed_points:
            _A[x, y] = 0
        del removed_points[:]

    output = _A

    return output


def TabelleLatex(elements):
    inputelements = []
    outputelements = []
    inputcount = 0
    outputcount = 0
    for element in elements:
        if type(element[0]) == Gates.Input:
            element[0].setFalse()
            inputelements.append(element)
            inputcount += 1
        elif type(element[0]) == Gates.Output:
            outputelements.append(element)
            outputcount += 1
    print("\\begin{tabular}{",end="")
    for _ in range(inputcount):
        print("|c",end="")
    print("x{2pt}",end="")
    for _ in range(outputcount):
        print("c|",end="")
    print("}")
    print("\\hline")
    for element in inputelements:
        print("\\textbf{In: " + str(element[1])+ "}", end="&")
    for element in outputelements:
        if element == outputelements[-1]:
            print("\\textbf{Out:" + str(element[1])+ "}", end="\\\\")
        else:
            print("\\textbf{Out:" + str(element[1]) + "}", end="&")
    print("")
    for element in inputelements:
        print(str(element[0].getValue()), end="&")
    for element in outputelements:
        if element == outputelements[-1]:
            print(str(element[0].getValue()), end="\\\\")
        else:
            print(str(element[0].getValue()), end="&")
    print("")
    for _ in range(1, int(math.pow(2, len(inputelements)))):
        print("\\hline")
        scounter = 0
        switched = True
        while switched and scounter < len(inputelements):
            switched = not inputelements[scounter][0].switch()
            scounter += 1
        for element in inputelements:
            print(str(element[0].getValue()), end="&")

        for element in outputelements:
            if element == outputelements[-1]:
                print(str(element[0].getValue()), end="\\\\")
            else:
                print(str(element[0].getValue()), end="&")
        print("")
    print("\\hline")
    print("\\end{tabular}")