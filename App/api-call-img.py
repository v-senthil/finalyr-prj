import requests
from pprint import pprint
url = 'http://localhost:5000/emotion/detect/image/api/v1'
my_img = {'image': open('D:\python3\Computer-Vision\Emotion-Detection\images\\happy\\happy1.jpg', 'rb')}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
pprint(r.json())
