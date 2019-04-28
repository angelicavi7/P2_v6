from flask import Flask
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from m09_model_deployment1 import predict_genre

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction',

ns = api.namespace('Predict', 
     description='Predict Genre')
   
parser = api.parser()

parser.add_argument('Plot',type=str,required=True,help='Plot',location='args')
parser.add_argument('Title',type=str,required=True,help='Title',location='args')
parser.add_argument('Year',type=int,required=True,help='Year',location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class MovieGenderApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_gender(args["Plot"],args["Title"],args["Year"])
        }, 200

if __name__ == '__main__':
    app.run(use_reloader=False, host='0.0.0.0', port=8888)