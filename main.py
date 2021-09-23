import os
import cv2
import time

# Define cores para fazer as marcacoes de objeto
COLORS = [(0,255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Recuperacao dos objetos treinados
class_names = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture('./videos/video.mp4')

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

foundAnimal = False

animalTime = 0
animalTurns = 0
start = 0

while True:
    _, frame = cap.read()

    # class -> objeto identificado
    # score -> certeza de identificacao
    # box -> onde achou objeto
    classes, scores, boxes = model.detect(frame, 0.1, 0.2) # Valores para melhorar rede neural
    for (classId, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classId) % len(COLORS)]

        label = f"{class_names[classId[0]].capitalize()} : {score}"

        if foundAnimal:
            animalTurns = animalTurns + 1

        if class_names[classId[0]] == 'cat' or class_names[classId[0]] == 'dog':
            if not foundAnimal:
                foundAnimal = True
                start = time.time()

            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Verifica se deve ativar ou nao a maquina
        # if 5 <= time.time() - start <= 5.1 and start != 0:
        #     if animalTurns >= 100:
        #         print(f'Tempo: {time.time() - start} | Turnos: {animalTurns} | Messagem: maquina ativada!')
        #     else:
        #         print(f'Tempo: {time.time() - start} | Turnos: {animalTurns} | Messagem: maquina nao ativada!')
        #
        #     foundAnimal = False
        #     animalTurns = 0
        #     start = 0

    cv2.imshow('Detections', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
