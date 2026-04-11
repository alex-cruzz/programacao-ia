#Import das libs
import cv2
from deepface import DeepFace
import time

#Passo 1: Carregar a identidade (cadastramento)
imagem_referencia = "face_id.jpg"
print("Carregando identida do moradir. ")

#Pré-análise da imagem, pra garantir que foto de referência é válida
try:
    DeepFace.represent(img_path = imagem_referencia, model_name = "VGG-Face")
    print("Identidade carregada com sucesso!")
except:
    print("Erro! Não encontrei o arquivo ou não há rosto nele.")
    exit()

#Iniciar a câmera
cap = cv2.VideoCapture(0) #O número 0 indica que a câmera está integrada ao computador
print("Sistema de portaria ativo.")

while True:
    ret, frame = cap.read() #ret retorna true se a foto foi tirada, frame recebe a imagem
    if not ret: break

    frame_small = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)

    #Desenhar um retângulo pra indicar a área de leitura
    height, width, _ = frame.shape
    cv2.rectangle(frame, (100,100), (width - 100, height - 100), (255,0,0),2)
    #Tamanho, cor e espessura da linha

    #Verificação da imagem com o rosto detectado
    cv2.putText(frame, "Pressione V para verificar o acesso", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('v'):
        print("Verificando identidade")
        try:
            resultado = DeepFace.verify(
                img1_path = frame, #quem está na câmera
                img2_path = imagem_referencia, #foto capturada
                model_name = "VGG-Face",
                enforce_detection = False
            )
            #Se o resultado é verdadeiro
            if resultado ['verified']:
                print(">>>>ACESSO LIBERADO<<<<")
                cv2.rectangle(frame, (0,0), (width, height), (0,255,0),2)
                cv2.imshow("Portaria", frame)
                cv2.waitKey(2000) #Pausa por 2s para mostrar a borda verde
            else:
                print(">>>>ACESSO NEGADO<<<<")
                cv2.rectangle(frame,(0,0),(width, height), (255,0,0),2)
                cv2.imshow("Portaria",frame)
                cv2.waitKey(2000)
        
        except Exception as e:
                print(f"Erro na leitura: {e}")
        
    cv2.imshow("Portaria", frame)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()