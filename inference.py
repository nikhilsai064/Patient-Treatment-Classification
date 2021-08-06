import pickle
import numpy as np
import pandas as pd
sc = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svc_1', 'rb'))
class_names = [0, 1]
def predict(df):
    df = df[['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']]
    jk = {"M": 1, "F": 0}
    df["SEX"] = df["SEX"].map(jk)
    numpy_array = np.array(df)
    numpy_array1 = sc.transform(numpy_array)
    predictions = model.predict(numpy_array1)
    test_pred = model.predict_proba(numpy_array1)
    output = [class_names[class_predicted] for class_predicted in predictions]
    for i in range(len(output)):
        con = test_pred[i]
        con1 = [round(con[0] * 100, 1), round(con[1] * 100, 1)]
        if (output[i] == 0):
            res = print("The Patient can be Out Cared. The confidence is ", max(con1), "%")
        else:
            res = print("The Patient can be In  Cared. The confidence is ", max(con1), "%")
    return res

