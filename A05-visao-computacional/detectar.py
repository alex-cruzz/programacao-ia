#Import das bibliotecas
import cv2 #opencv -> lib responsável pelo gerenciamento de dispositivos vis. comp.
from ultralytics import YOLO #Responsável pelo reconhecimento facial/objetos.

#Passo 1: carregamento de modelo
print("Carregando modelo...")
model = YOLO('yolov8n.pt') #Yolov8n é uma versão nano, mais leve/rápida

#Abrir uma conexão com webcam 
cap = cv2.VideoCapture(0)
#O número 0 representa uma webcam integrada ao computador
#O número 1 representa uma webcam conectada via USB (via física)
#Caso a via seja remota, o endereço de Ip deve ser informado

#Verifica se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar câmera")
    exit()

print("Iniciando detcção. Pressione 'q' para sair")

#Passo 3: Inicar a leitura das detecções
while True:
    sucesso, frame = cap.read() #Ler os frames da câmera

    if sucesso: #Realizar a detecção (inference)
        results = model(frame, conf = 0.5) #Queremos detecções 50% ou mais de certeza
        annotated_frame = results[0].plot() #Criar caixa virtual da imagem
        cv2.imshow("Visão Computacional - YOLOv8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #Pressionar "q" para sair
            break

    else:
        break    

#Limpeza
cap.release()
cv2.destroyAllWindows()