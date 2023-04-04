import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))

@app.route('/')
def index():
    return render_template('mushrooms.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        column_list = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat']
        column_category_lists = [['bell','conical','convex','flat','knobbed','sunken'],['fibrous','grooves','scaly','smooth'],
        ['brown','buff','cinnamon','gray','green','pink','purple','red','white','yellow'],['bruises','no'],['almond','anise','creosote','fishy','foul','musty','none','pungent','spicy'],
        ['attached','descending','free','notched'],['close','crowded','distant'],['broad','narrow'],['black','brown','buff','chocolate','gray','green','orange','pink','purple','red','white','yellow'],
        ['enlarging','tapering'],['bulbous','club','cup','equal','rhizomorphs','rooted','missing'],['fibrous','scaly','silky','smooth'],['fibrous','scaly','silky','smooth'],
        ['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],['brown','buff','cinnamon','gray','orange','pink','red','white','yellow'],['brown','orange','white','yellow'],['none','one','two'],
        ['cobwebby','evanescent','flaring','large','none','pendant','sheathing','zone'],['black','brown','buff','chocolate','green','orange','purple','white','yellow'],
        ['abundant','clustered','numerous','scattered','several','solitary'],['grasses','leaves','meadows','paths','urban','waste','woods']]
        counter = 0
        features = []
        for i in column_list:
            value = request.form.get(i)
            if value in column_category_lists[counter]:
                features.append(value)
                counter = counter + 1
            else:
                break

        features1 = [x for x in request.form.values()]
        feature_array = np.array(features).reshape(1,-1)
        feature_array_df = pd.DataFrame(feature_array, columns = column_list)
        final_output = transformer.transform(feature_array_df)
        prediction  = model.predict(final_output)
        print(prediction)
        
        if prediction == 1:
            return render_template('mushrooms.html', pred='This is a Poisonous Mushroom')
        else:
            return render_template('mushrooms.html', pred='This is an Edible Mushroom')
    except:
        return render_template('mushrooms.html', pred='INVALID INPUT')

if __name__=='__main__':
    app.run(debug=True)
