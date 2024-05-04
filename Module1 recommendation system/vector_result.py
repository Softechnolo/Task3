from flask import Flask, request, jsonify
from pinecone import Pinecone

app = Flask(__name__)
pc = Pinecone(api_key="a2d61230-ed5f-4887-9107-53c455b85b68")
index = pc.Index(index_name="product-recommendation")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data['query']
    
    # Query the Pinecone vector index
    results = index.query(queries=[query], top_k=5)
    
    # Return the results
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
