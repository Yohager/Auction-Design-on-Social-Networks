import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import random 
import types

class IDMVerify:

    def __init__(self, graph, seller_id, buyer_ids) -> None:
        '''
        graph: networked market 
        seller_id: node id of the seller 
        buyer_ids: node id of buyers 
        '''
        self.graph = graph 
        self.cnt = len(self.graph.nodes()) 
        self.seller = seller_id
        self.buyers = buyer_ids

    def ValGeneration(self, distribution, lower_bound, upper_bound, num):
        '''
        generating bids under specific distribution 
        '''
        if distribution == 'uniform':
            return np.random.uniform(lower_bound, upper_bound, num)
        
    
    def PlotGraph(self) -> None:
        nx.draw(self.graph, with_labels = True)

    def _ConstructDS(self) -> dict:
        '''
        calculate the dominant set of each buyer 
        '''
        principal = 0
        node_set = set(self.graph.nodes)
        agent_set = node_set - {principal}
        dominating_dict = {contestant: set() for contestant in agent_set}
        leaf_set = set()
        for node in agent_set:
            sub_graphs = self.graph.subgraph([i for i in node_set if i != node])
            connected_parts = list(nx.connected_components(sub_graphs))
            if len(connected_parts) != 1:
                for part in connected_parts:
                    if principal not in part:
                        dominating_dict[node] = dominating_dict[node] | part
        return dominating_dict

    def _CheckAllocation(self, idx, next_idx, bids, next_idx_dominate):
        '''
        idx: the id of current bidder 
        next_idx: the next on sequence bidder 
        bids: dict of all bidders 
        d_idx: bidders dominated by idx 
        '''        
        if not next_idx:
            # the bidder with highest bid 
            return True
        cutting_set = {next_idx} + set(next_idx_dominate)
        
        


    def IDM(self, vals):
        '''
        run IDM directly 
        calcuate the allocation and payment 
        vals: bidders valuation 
        market is fixed as the self.graph 
        '''
        # locate the highest bid
        d = {k:v for k, v in zip(self.buyers, vals)} 
        m = list(d.items())
        m.sort(key = lambda x : -x[1])
        if not m:
            raise ValueError('no buyer exists!')
        highest_bidder = m[0][0]
        # find all the nodes dominating the highest bidder and rank them a sequence
        dominating_dict = self._ConstructDS()
        dominate_bidders = {}
        for k, v in dominating_dict.items():
            if highest_bidder in v:
                dominate_bidders.add(k)
        dominate_bidders_dists = []
        for bidder in dominate_bidders:
            cur_dist = nx.shortest_path_length(self.graph, self.seller, bidder)
            dominate_bidders_dists.append(cur_dist, bidder)
        dominate_bidders_dists.sort(key = lambda x : x[0])
        winner = None 
        for i, bidder in enumerate(dominate_bidders):
            if bidder == highest_bidder:
                winner = bidder 
                break 


    
    def IDMRev(self, vals):
        '''
        calculate revenue of IDM directly: Rev = SW*(N\d1)
        '''
        m = list(zip(self.buyers, vals))
        m.sort(key = lambda x : -x[1])
        if not m:
            raise ValueError('no buyer exists!')
        highest_bidder = m[0][0]
        # print('profile', m)
        dominante_d = self._ConstructDS(self.graph)
        critical_nodes = nx.shortest_path(self.graph, self.seller, highest_bidder)
        # print(critical_nodes)
        first_critical_node = critical_nodes[1]
        # print('first', first_critical_node)
        left_nodes = set(self.graph.nodes) - dominante_d[first_critical_node]
        left_nodes = left_nodes - {self.seller}
        left_nodes = left_nodes - {first_critical_node}
        # print('left_nodes', left_nodes)
        # max_second_price = -float('inf')
        left_vals = []
        for node in left_nodes:
            for idx, v in m:
                if idx == node:
                    left_vals.append(v)
        return max(left_vals)     

    def main(self, iterations):
        self.PlotGraph()
        total_rev = 0 
        for _ in range(iterations):
            vs = self.ValGeneration('uniform', 0, 1, self.cnt - 1)
            total_rev += self.IDMRev(vs)
        return total_rev / iterations
    

if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_nodes_from([0,1,2,3,4,5])
    G1.add_edges_from([(0,1), (0,2), (0,3), (0,4), (4,5)])

