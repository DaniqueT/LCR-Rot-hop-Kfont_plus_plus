import torch.nn as nn
from rdflib import URIRef, Graph
import torch
import gensim
from .act_fun import gelu
import math
from rdflib.namespace import RDFS, RDF, OWL



class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size, layer, currentLayerIndex, dense1, dense2, proj1, proj2, word2vec, ontology):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.currentLayerIndex = currentLayerIndex
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Change depending on in which layer the knowledge gets added
        if self.currentLayerIndex in range(9,12):
            self.word2vec_model = word2vec
            self.ontology = ontology
            self.synonym_vectors = {}

        self.dense1 = dense1
        self.dense2 = dense2
        self.projection_matrix_1 = proj1
        self.projection_matrix_2 = proj2

    def forward(self, x, sentence, knowledge_layers):
        inter = self.linear_1(x)

        if self.currentLayerIndex in knowledge_layers and sentence is not None:
            knowledge = self.collect_projectedknowledge(sentence = sentence)
            know1 = self.projection_matrix_1(knowledge)
            know1 = know1.unsqueeze(0)
            expanded_states = torch.cat([inter, know1], dim=-1)
            inter = self.dense1(expanded_states)

        inter = gelu(inter)
        output = self.linear_2(inter)

        if self.currentLayerIndex in knowledge_layers and sentence is not None:
            know2 = self.projection_matrix_2(knowledge)
            know2 = know2.unsqueeze(0)
            expanded_output = torch.cat([output, know2], dim=-1)
            output = self.dense2(expanded_output)
            output = self.layer_norm(output)

        return output

    def collect_projectedknowledge(self, sentence, max_hops=2, gamma=0.5):
        knowledge_vectors = []

        for word in sentence:
            uri = self.find_uri_for(lex=word, ontology=self.ontology)

            if uri:
                hop_vectors = []

                for hop in range(max_hops + 1):

                    if hop == 0:
                        # synonyms
                        concepts = self.find_synonyms_for(uri, self.ontology)
                      
                    else:
                        # k-hop neighbours
                        concepts = self.find_khop_concepts(uri, hop)
                       
                    if concepts is not None and len(concepts) > 0:
                        vec = self.get_synonym_vectors(concepts)

                        # weight neighbours less
                        weight = 1.0 if hop == 0 else math.exp(-(hop + gamma))
                        hop_vectors.append(weight * vec)

                if hop_vectors:
                    vector = torch.sum(torch.cat(hop_vectors, dim=0), dim=0, keepdim=True)
                    knowledge_vectors.append(vector)
                else:
                    knowledge_vectors.append(torch.zeros(1,100))

            else:
                knowledge_vectors.append(torch.zeros(1,100))

        knowledge_vectors = torch.cat(knowledge_vectors).to(self.device)
        return knowledge_vectors

    def get_synonym_vectors(self, synonyms):
        vectors = []
        model = self.word2vec_model

        for synonym in synonyms:
            if synonym in model.wv.index_to_key:
                iri_vector = model.wv.get_vector(synonym)
                vectors.append(torch.tensor(iri_vector))

        if vectors:
            vectors = torch.stack(vectors)
            vector = torch.mean(vectors, dim=0, keepdim=True)
            return vector

        return torch.zeros(1,100)
    
    def find_khop_concepts(self, uri, k):

        visited = set([uri])
        current_level = [uri]

        for _ in range(k):
            next_level = []

            for node in current_level:

                for _, _, obj in self.ontology.triples((node, RDFS.subClassOf, None)):
                    if obj not in visited:
                        visited.add(obj)
                        next_level.append(obj)

                for _, _, obj in self.ontology.triples((node, OWL.equivalentClass, None)):
                    if obj not in visited:
                        visited.add(obj)
                        next_level.append(obj)

            current_level = next_level

        concepts = []
        for concept in current_level:
            lex = self.find_synonyms_for(concept, self.ontology)
            if lex:
                concepts.extend(lex)

        return concepts

    @staticmethod
    def find_synonyms_for(resource: URIRef, ontology: Graph) -> list[str]:
        NAMESPACE = "http://www.kimschouten.com/sentiment/restaurant"
        lex = [str(item[2]) for item in ontology.triples((resource, URIRef("#lex", NAMESPACE), None))]
        return lex

    @staticmethod
    def find_uri_for(lex: str, ontology: Graph) -> URIRef | None:
        if lex == '"':
            return None

        result = ontology.query(f"""
                    SELECT ?subject
                    {{ ?subject restaurant1:lex "{lex}" }}
                    LIMIT 1
                    """)
        for row in result:
            return row.subject
        return None