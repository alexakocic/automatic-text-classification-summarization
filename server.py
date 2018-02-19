from lov_train_classify import *
import http.server
from urllib.parse import urlparse
from urllib.parse import parse_qs

normalizer = None
pipeline = None
labels = None
vectorizer = None
modified_vectorizers = None
bulk_classifier = None 
label_mappings = None

class Server(http.server.BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parsed_url.query
        self._set_headers()
        
        if path == '/train':
            filepath = parse_qs(query)['path'][0]
            print(filepath)
            
            normalizer, pipeline, labels, \
            vectorizer, modified_vectorizers, main_classifier, \
            label_mappings = train(filepath, 'path')
            
            self.wfile.write(b"Training has Finished.")
        elif path == '/classify':
            try: 
                text = parse_qs(query)['text'][0]
                result = classify(text, normalizer, pipeline, labels, vectorizer, modified_vectorizers, bulk_classifier, label_mappings)
                self.wfile.write(str.encode(str(result)[1: len(str(result)) - 1]))
            except Exception:
                self.wfile.write("Error while classifying. Please train model first.")

server = http.server.HTTPServer(('localhost', 8090), Server)
print("Starting server...")
server.serve_forever()