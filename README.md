# VisaoComputacional

# Descrição do Projeto: 
    O Projeto realiza o treinamento de um algoritmo de machine learning e classifica os objetos com o TeachableMachine, e,
no código em python, resultante do treinamento, com funções de reconhecimento de faces e rótulos de
classificação utilizando a biblioteca OpenCV. 

# Instruções de Instalação:
   No Anaconda, abra o Jupyter, tenha as bibliotecas Kera e , vá ao site da aplicação TeachableMachine: abrir novo projeto de imagem no modelo padrão, com o app Irium, criar o banco de imagens com 5 objetos, utilizando uma webcam (aproximadamente 200 imagens de cada item).

# Instruções de Uso:
  Utilizar o aplicativo Irium no seu celular. O computador e o celular devem estar na
mesma rede WIFI (criar um ponto de acesso com o celular). Se não puder realizar com o Irium, use uma
webcam diretamente ligada ao computador.
Alterar e Incrementar no código o seguinte:
10.1) Corrigir a entrada da câmera para 1 com Irium, e 0 para webcam.
# CAMERA pode ser 0 ou 1 com base na câmera padrão do seu computador
camera = cv2.VideoCapture(1)
10.2) Incluir o classificador Haar Cascade para identificar faces no vídeo, antes do laço WHILE TRUE
# Carregue o classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
10.3) Dentro do laço WHILE TRUE, substituir pelo seguinte:
*Reparar nas indentações dentro dos laços FOR e do WHILE TRUE
# Captura a imagem da câmera
ret, image = camera.read()
# Detecta faces na imagem
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# Loop pelas faces detectadas
for (x, y, w, h) in faces:
# Extrai a região da face da imagem
face_roi = image[y:y + h, x:x + w]
# Redimensiona a imagem da face para o tamanho necessário para o modelo
face_roi = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
# Converte a imagem da face em um array numpy e aplica normalização
face_array = np.asarray(face_roi, dtype=np.float32).reshape(1, 224, 224, 3)
face_array = (face_array / 127.5) - 1
# Faz a previsão usando o modelo
prediction = model.predict(face_array)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]
# Desenha o retângulo na imagem
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Retângulo verde
cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Nome da pessoa
# Imprime a previsão e a pontuação de confiança
print("Pessoa:", class_name)
print("Pontuação de Confiança:", str(np.round(confidence_score * 100))[:-2], "%")
# Mostra a imagem na janela
cv2.imshow("Webcam Image", image)
# Escuta o teclado para interrupção
keyboard_input = cv2.waitKey(1)
# 27 é o código ASCII para a tecla Esc
if keyboard_input == 27:
break
 Rodar o código utilizando a câmera do celular com app Irium
Apontar a câmera para cada objeto, tirar um print da tela e salvar o print.
Apontar a câmera para dois objetos e até identificar os dois objetos diferentes na mesma imagem; tirar um
print da tela e salvar o print.
                   
# Créditos: Contribuidores e quaisquer outras fontes ou recursos utilizados (ex: realizado no laboratório de
Sistemas de Informação da UFOPA).
Foi realizado no laboratório de Sistema de Informação pelos discentes de sistema Anne, Neidiane e Nivaldo com o auxilio do professor Danilo Figueiredo.
