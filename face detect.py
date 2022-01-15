import cv2

video = cv2.VideoCapture(0)
## Classificador
classificadorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
## Loop
while True:
    ## mantendo conectada a webcam e os frames
    conectado, frame = video.read()
    ## transformando em escala de cinza
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ## detectando a face com o classificador treinado xml
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza,minSize=(20,20))
    for (x,y,l,a) in facesDetectadas:
        cv2.rectangle(frame, (x,y), (x+l, y+a), (0,0,255), 2)
    
    cv2.imshow('Video',frame)
    
    ## Fechar janela do video
    if cv2.waitKey(1) == ord('s'):
        break
      
## Liberar a captura
video.release()
## Liberar memória
cv2.destroyAllWindows()