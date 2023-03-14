import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import random 
import types
import collections 
from tqdm import tqdm

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

    def _AllocationAndPayment(self, idx, next_idx, bids, idx_dominate, next_idx_dominate):
        '''
        idx: the id of current bidder 
        next_idx: the next on sequence bidder 
        bids: dict of all bidders 
        d_idx: bidders dominated by idx 
        '''  
        # market N\i and market N\d_i 
        idx_cutting_set = {*{idx}, *idx_dominate} # cutting nodes set for bidder i 
        next_idx_cutting_set = None 
        # print('here', idx, next_idx, bids, idx_dominate, next_idx_dominate)
        if next_idx:
            next_idx_cutting_set = {*{next_idx}, *next_idx_dominate} # cutting nodes set for bidder i + 1 
        max_v_without_idx = -float('inf')
        for bidder in self.buyers:
            if bidder not in idx_cutting_set:
                max_v_without_idx = max(max_v_without_idx, bids[bidder])
        # print('max_v_without_idx', max_v_without_idx)
        if not next_idx:
            return True, max_v_without_idx
        else:
            max_v_without_next_idx = -float('inf')
            for bidder in self.buyers:
                if bidder not in next_idx_cutting_set:
                    max_v_without_next_idx = max(max_v_without_next_idx, bids[bidder])
            # print('max_v_without_next_idx', max_v_without_next_idx)
            if bids[idx] == max_v_without_next_idx:
                # print('here')
                return True, max_v_without_idx
            else:
                return False, max_v_without_idx - max_v_without_next_idx 

        
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
        dominate_bidders = set()
        for k, v in dominating_dict.items():
            if highest_bidder in v:
                dominate_bidders.add(k)
        dominate_bidders_dists = []
        for bidder in dominate_bidders:
            cur_dist = nx.shortest_path_length(self.graph, self.seller, bidder)
            dominate_bidders_dists.append([cur_dist, bidder])
        dominate_bidders_dists.sort(key = lambda x : x[0])
        dominate_sequence = [x[1] for x in dominate_bidders_dists] + [highest_bidder]
        # print(dominate_sequence)
        winner = collections.defaultdict(int) 
        rewarded_bidders = collections.defaultdict(int)
        for i, bidder in enumerate(dominate_sequence):
            if i == len(dominate_sequence) - 1:
                # IDM mechanism visits the bidder with highest bid 
                _, payment = self._AllocationAndPayment(bidder, None, d, dominating_dict[bidder], None)
                winner[bidder] = payment 
            else:
                flag, payment = self._AllocationAndPayment(bidder, dominate_sequence[i+1], d, dominating_dict[bidder], dominating_dict[dominate_sequence[i+1]])
                if flag:
                    # current bidder is the winner in IDM mechanism 
                    # payment is the highest bid in N\i 
                    winner[bidder] = payment 
                    break 
                else:
                    # current bidder is a loser in IDM mechanism 
                    rewarded_bidders[bidder] = payment
        rev = sum(winner.values())
        rev += sum(rewarded_bidders.values())
        return winner, rewarded_bidders, rev 

    
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
        dominante_d = self._ConstructDS()
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
        run_rev = 0
        for _ in tqdm(range(iterations)):
            vs = self.ValGeneration('uniform', 0, 1, self.cnt - 1)
            total_rev += self.IDMRev(vs)
            _, _, rev = self.IDM(vs)
            run_rev += rev 
        return total_rev / iterations, run_rev / iterations
    

if __name__ == "__main__":
    G1 = nx.Graph()
    G1.add_nodes_from([0,1,2,3,4,5])
    G1.add_edges_from([(0,1), (0,2), (0,3), (0,4), (4,5)])
    G2 = nx.Graph()
    G2.add_nodes_from([0,1,2,3,4,5])
    G2.add_edges_from([(0,1), (0,2), (0,3), (2,4), (3,5)])
    G3 = nx.Graph()
    G3.add_nodes_from([0,1,2,3,4,5])
    G3.add_edges_from([(0,1), (0,2), (0,3), (3,4), (4,5)])
    G4 = nx.Graph()
    G4.add_nodes_from([0,1,2,3,4,5])
    G4.add_edges_from([(0,1), (0,2), (2,3), (2,4), (4,5)])
    seller_id = 0
    buyer_ids = [1,2,3,4,5]
    IDM_test = IDMVerify(G4, seller_id, buyer_ids)
    print(IDM_test.main(1000))