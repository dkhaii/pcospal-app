import tensorflow as tf
from tensorflow import lite as tflite

input_data = {}
age = int(input("Your age: "))
input_data['age'] = age

bmi = float(input("Please enter your Body Mass Index (BMI): "))
input_data['bmi'] = bmi

pulse_rate = int(input("What is your current pulse rate? "))
input_data['pulse_rate'] = pulse_rate

marriage_yrs = int(input("How long you been married? "))
input_data['marriage_yrs'] = marriage_yrs

hip = int(input("Please enter your current hip size: "))
input_data['hip'] = hip

waist = int(input("Please enter your current waist size: "))
input_data['waist'] = waist

cycle = input("Is your menstrual cycle regular or irregular? ")
if cycle == 'regular':
    cyle = 2
    input_data['cycle'] = cycle
elif cycle == 'irregular':
    cycle = 4
    input_data['cycle'] = cycle

weight_gain = input("Are you experienced weight gain (y/n)? ")
if weight_gain == 'y':
    weight_gain = 1
    input_data['weight_gain'] = weight_gain
elif  weight_gain == 'n':
    weight_gain = 0
    input_data['weight_gain'] = weight_gain

hair_growth = input("Are you experienced hair growth (y/n)? ")
if hair_growth == 'y':
    hair_growth = 1
    input_data['hair_growth'] = hair_growth
elif hair_growth == 'n':
    hair_growth = 0
    input_data['hair_growth'] = hair_growth

skin_darkening = input("Are you experienced skin darkening (y/n)? ")
if skin_darkening == 'y':
    skin_darkening = 1
    input_data['skin_darkening'] = skin_darkening
elif  skin_darkening == 'n':
    skin_darkening = 0
    input_data['skin_darkening'] = weight_gain

hair_loss = input("Are you experienced hair loss (y/n)? ")
if hair_loss == 'y':
    hair_loss = 1
    input_data['hair_loss'] = hair_loss
elif hair_loss == 'n':
    weight_gain = 0
    input_data['hair_loss'] = weight_gain

pimples = input("Are you experienced pimples (y/n)? ")
if pimples == 'y':
    pimples = 1
    input_data['pimples'] = pimples
elif pimples == 'n':
    pimples = 0
    input_data['pimples'] = pimples

fast_food = input("Do you often eat fast food (y/n)? ")
if fast_food == 'y':
    fast_food = 1
    input_data['fast_food'] = fast_food
elif fast_food == 'n':
    fast_food = 0
    input_data['fast_food'] = fast_food

interpreter = tflite.Interpreter(model_path='pcos-model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for name, value in input_data.items():
    if name == 'bmi':
        input_data[name] = tf.convert_to_tensor([[value]], dtype=tf.float32)
    else:
        input_data[name] = tf.convert_to_tensor([[value]], dtype=tf.int64)

for input_name, input_tensor in input_data.items():
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    input_tensor = tf.cast(input_tensor, dtype=input_dtype)
    interpreter.set_tensor(input_index, input_tensor)

interpreter.invoke()

print("Predicting...")
output_data = interpreter.get_tensor(output_details[0]['index'])

prediction = 0 if output_data[0][0] < 0.5 else 1

if prediction == 0:
    print("Based on your condition, you are at LOW RISK of experiencing Polycystic Ovary Syndrome (PCOS)")
else:
    print("Based on your condition, you are at HIGH RISK of experiencing Polycystic Ovary Syndrome (PCOS)")

