"""Two independent knowledge branches.

Branch A (documents) and Branch B (web) are searched in parallel and
kept separate all the way to the prompt: the model sees a document
section and a web section, never one blended pile of evidence.
"""

from backend.application.knowledge.documents import DocumentKnowledgeBranch
from backend.application.knowledge.web import WebKnowledgeBranch

__all__ = ["DocumentKnowledgeBranch", "WebKnowledgeBranch"]
