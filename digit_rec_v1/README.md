You can use this code to recognize handwritten digits. 

Before start, you should download packages in requirement.txt.

if you have finished downloading, then it is time to start.

Firstly, RUN model.py to generate a model ( .h5 )  to recognize single digit, which has actually already generated，whose name is my_model.h5.

Before fun load_image, put your image into 'test_fig' directory, and change the path of image:

```python
img = cv2.imread('test_fig/no_order.png')
```



RUN load_image.py to load data of your test image.

RUN predict.py and the predicted number will be shown on you terminal.





