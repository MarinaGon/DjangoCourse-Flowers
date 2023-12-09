from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
import tensorflow as tf
import os

classes = ['астильба', 'колокольчик', 'черноглазая_сьюзен', 'календула', 'калифорнийский_мак', 'гвоздика',
           'обыкновенная ромашка', 'кореопсис', 'нарцисс', 'одуванчик', 'ирис', 'магнолия', 'роза', 'подсолнух',
           'тюльпан', 'кувшинка']

model = tf.keras.models.load_model('djangoProjectMLLab/flowers.h5')

def upload_file(request):
    if request.method == 'POST' and request.FILES:
        file = request.FILES['myfile']
        upload_folder = os.path.join('djangoProjectMLLab', 'uploaded_images')
        fs = FileSystemStorage(location=upload_folder)

        filename = fs.save(file.name, file)
        file_url = fs.url(filename)


        size = (224, 224)
        image = Image.open(file)
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array).flatten()
        predictions = tf.where(predictions < 0.5, 0, 1)
        predicted_class_index = tf.argmax(predictions)

        res_message = f'Скорее всего на этом фото {classes[predicted_class_index]}'

        return render(request, 'upload_file.html', {
            'res_message': res_message,
        })
    return render(request, 'upload_file.html')
