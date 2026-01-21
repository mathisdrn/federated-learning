from fluke.algorithms import CentralizedFL
from fluke.server import Server
from src.fairness.client import FairClient


class FairFedAVG(CentralizedFL):
    def get_client_class(self):
        return FairClient

    def get_server_class(self):
        return Server
