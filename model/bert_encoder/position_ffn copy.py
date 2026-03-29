import torch.nn as nn
from rdflib import URIRef, Graph
import torch
#import gensim
from .act_fun import gelu


import torch
import torch.nn as nn
from rdflib import URIRef, Graph
from .act_fun import gelu
import math
from rdflib.namespace import RDFS, RDF, OWL


    
class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size, layer, currentLayerIndex, dense1, dense2, proj1, proj2, ontology, word2vec, knowledge_layers):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.currentLayerIndex = currentLayerIndex
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Change depending on in which layer the knowledge gets added & change name of paths
        #if self.currentLayerIndex in range(9,12):
            #self.synonym_vectors = {}
            #self.path_owl = "data/myontology2016/output"

        self.dense1 = dense1
        self.dense2 = dense2
        self.projection_matrix_1 = proj1
        self.projection_matrix_2 = proj2
        if self.currentLayerIndex in knowledge_layers:
            self.ontology = ontology
            self.synonym_vectors = {}
            self.word2vec_model = word2vec
            self.synonym_embeddings = torch.tensor(word2vec.wv.vectors, dtype=torch.float32, device=self.device)
            self.word_to_idx = {word: i for i, word in enumerate(word2vec.wv.index_to_key)}
            self.random_embeddings = torch.randn_like(self.synonym_embeddings)
            self.inject_knowledge_during_training = True
            self.lex_to_uri = self.build_lex_to_uri(self.ontology)
        

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
    
    def collect_projectedknowledge(self, sentence, max_hops=1, gamma=0.5):
        """
        Collect weighted knowledge embeddings for a sentence.
        Handles BERT wordpieces by reconstructing full words.
        max_hops: 0 = only synonyms, 1 = include 1-hop neighbors, etc.
        """
        knowledge_vectors = []

        # Step 1: Reconstruct full words
        full_words = []
        token_to_word_idx = []  # map each token to its full word index
        buffer = ""
        word_idx = -1

        for token in sentence:
            if token in ["[CLS]", "[SEP]"]:
                full_words.append(token)
                word_idx += 1
                token_to_word_idx.append(word_idx)
                continue

            if token.startswith("##"):
                buffer += token[2:]
                token_to_word_idx.append(word_idx)  # same word as previous
            else:
                if buffer:  # save previous word
                    full_words.append(buffer)
                buffer = token
                word_idx += 1
                token_to_word_idx.append(word_idx)

        if buffer:
            full_words.append(buffer)
        
        # Step 2: Collect knowledge vectors per full word
        full_word_vectors = []
        for word in full_words:
            word_vector = torch.zeros(1, self.synonym_embeddings.size(1), device=self.device)

            if word in ["[CLS]", "[SEP]"]:
                full_word_vectors.append(word_vector)
                continue

            uri = self.find_uri_for(word)
            if uri:
                all_hops_vectors = []
                for hop in range(max_hops + 1):
                    if hop == 0:
                        # 0-hop: synonyms (skip the word itself)
                        concepts = self.find_synonyms_for(uri, self.ontology)
                        concepts = [c for c in concepts if c.lower() != word.lower()]
                    else:
                        # i-hop: traverse ontology
                        concepts = self.find_khop_concepts(uri, hop)
                    if concepts:
                        hop_avg = self.get_synonym_vectors(concepts)
                        weight = 1.0 if hop == 0 else math.exp(-(hop + gamma))
                        all_hops_vectors.append(weight * hop_avg)

                if all_hops_vectors:
                    word_vector = torch.sum(torch.cat(all_hops_vectors, dim=0), dim=0, keepdim=True)

            full_word_vectors.append(word_vector)

        # Step 3: Map knowledge vectors back to original tokens
        for idx in range(len(sentence)):
            word_idx = token_to_word_idx[idx]
            knowledge_vectors.append(full_word_vectors[word_idx])

        knowledge_vectors = torch.cat(knowledge_vectors, dim=0).to(self.device)
        return knowledge_vectors

    
    
    def get_synonym_vectors(self, synonyms):
        vectors = []

        for synonym in synonyms:
            if synonym in self.word_to_idx:
                idx = self.word_to_idx[synonym]
                if self.inject_knowledge_during_training:
                    iri_vector = self.synonym_embeddings[idx]  # use trained embeddings
                else:
                    iri_vector = self.random_embeddings[idx] 
                #iri_vector = self.synonym_embeddings[idx]  # <-- use the learnable parameter
                vectors.append(iri_vector.unsqueeze(0))    # keep batch dimension

        if vectors:
            vectors = torch.cat(vectors, dim=0)
            vector = torch.mean(vectors, dim=0, keepdim=True)
            return vector

        return torch.zeros(1, self.synonym_embeddings.size(1), device=self.device)


    # def get_synonym_vectors(self, synonyms):
    #     vectors = []
    #     model = self.word2vec_model

    #     for synonym in synonyms:
    #         #if synonym in model.wv.index_to_key:
    #         if synonym in model.wv:
    #             iri_vector = model.wv.get_vector(synonym)
    #             vectors.append(torch.tensor(iri_vector))
             
    #     if vectors:
    #         vectors = torch.stack(vectors)
    #         vector = torch.mean(vectors, dim=0, keepdim=True)
    #         return vector

    #     return torch.zeros(1,100)

    def find_khop_concepts(self, uri, k):

        visited = set([uri])
        current_level = [uri]

        for _ in range(k):
            next_level = []

            for node in current_level:

                # follow subclass relations
                for _, _, obj in self.ontology.triples((node, RDFS.subClassOf, None)):
                    if obj not in visited:
                        visited.add(obj)
                        next_level.append(obj)

                # follow equivalent class relations
                for _, _, obj in self.ontology.triples((node, OWL.equivalentClass, None)):
                    if obj not in visited:
                        visited.add(obj)
                        next_level.append(obj)

            current_level = next_level

        # convert to lexical words
        concepts = []
        for concept in current_level:
            lex = self.find_synonyms_for(concept, self.ontology)
            concepts.extend(lex)

        return concepts

    # def find_khop_concepts(self, uri, k):
    #     """
    #     Traverse ontology k hops from the URI.
    #     Returns list of concept strings for embedding lookup.
    #     """
    #     visited = set()
    #     current_level = [uri]
    #     for _ in range(k):
    #         next_level = []
    #         for node in current_level:
    #             # get all classes connected to node
    #             neighbors = [str(o) for s,p,o in self.ontology.triples((node, None, None)) if o not in visited]
    #             next_level.extend(neighbors)
    #             visited.update(neighbors)
    #         current_level = next_level
    #     return current_level
    
    def build_lex_to_uri(self, ontology):
        """
        Precompute mapping from lex string → ontology URI.
        Avoids expensive SPARQL queries during training.
        """
        lex_to_uri = {}

        for s, p, o in ontology.triples((None, None, None)):
            if str(p).endswith("lex"):
                lex_to_uri[str(o)] = s


        return lex_to_uri
    
    def find_uri_for(self, lex: str):
        if lex == '"':
            return None
        return self.lex_to_uri.get(lex)
    
        
    def find_synonyms_for(self, resource, ontology):

        return [
            str(o)
            for s, p, o in ontology.triples((resource, None, None))
            if str(p).endswith("lex")
        ]

    # @staticmethod
    # def find_synonyms_for(resource: URIRef, ontology: Graph) -> list[str]:
    #     NAMESPACE = "http://www.kimschouten.com/sentiment/restaurant"
    #     lex = [str(item[2]) for item in ontology.triples((resource, URIRef("#lex", NAMESPACE), None))]
    #     return lex

    # @staticmethod
    # def find_uri_for(lex: str, ontology: Graph) -> URIRef | None:
    #     if lex == '"':
    #         return None

    #     result = ontology.query(f"""
    #                 SELECT ?subject
    #                 {{ ?subject restaurant1:lex "{lex}" }}
    #                 LIMIT 1
    #                 """)
    #     for row in result:
    #         return row.subject
    #     return None




