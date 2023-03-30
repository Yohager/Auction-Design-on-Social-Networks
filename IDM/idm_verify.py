import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import random 
import types
import collections 
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d 

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

    def _AllocationAndPaymentRP(self, idx, next_idx, bids, idx_dominate, next_idx_dominate, rp):
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
            return True, max(max_v_without_idx, rp) 
        else:
            max_v_without_next_idx = -float('inf')
            for bidder in self.buyers:
                if bidder not in next_idx_cutting_set:
                    max_v_without_next_idx = max(max_v_without_next_idx, bids[bidder])
            # print('max_v_without_next_idx', max_v_without_next_idx)
            if bids[idx] == max_v_without_next_idx:
                # print('here')
                if bids[idx] >= rp:
                    return True, max(max_v_without_idx, rp)
                else:
                    return False, max(max_v_without_idx, rp) - max(max_v_without_next_idx, rp) 
            else:
                return False, max(max_v_without_idx, rp) - max(max_v_without_next_idx, rp)  

       
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

    def IDM_with_rp(self, vals, rp):
        '''
        run IDM revenue result with reserved price 
        '''
        # locate the highest bid
        d = {k:v for k, v in zip(self.buyers, vals)} 
        m = list(d.items())
        m.sort(key = lambda x : -x[1])
        if not m:
            raise ValueError('no buyer exists!')
        highest_bidder = m[0][0]
        if m[0][1] < rp:
            return None, None, 0 # the highest bid is smaller than the reserved price 
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
                _, payment = self._AllocationAndPaymentRP(bidder, None, d, dominating_dict[bidder], None, rp)
                winner[bidder] = payment 
            else:
                flag, payment = self._AllocationAndPaymentRP(bidder, dominate_sequence[i+1], d, dominating_dict[bidder], dominating_dict[dominate_sequence[i+1]], rp)
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

    def IDMRev_RP(self, vals, rp):
        m = list(zip(self.buyers, vals))
        m.sort(key = lambda x : -x[1])
        if not m:
            raise ValueError('no buyer exists!')
        highest_bidder = m[0][0]
        # print('profile', m)
        if m[0][1] < rp:
            return 0 # no bidder has bid higher than rp 
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
        return max(max(left_vals), rp)     
    
    def optimal_r_theoretical(self, alpha, n, m):
        '''
        alpha: vector which contains all alpha_i
        n: total bidder number 
        m: all the branches number of the tree 
        '''
        c = collections.Counter(alpha)
        c_sort = [(k, v) for k, v in c.items()]
        c_sort.sort(lambda x: -x[1])
        coefs = [x[1] for x in c_sort] + [-1 * (m + n)]
        roots = np.roots(coefs)
        update_roots = [1 / x for x in roots]
        for ur in update_roots:
            if 0 <= ur <= 1:
                return ur 
        raise ValueError('No Optimal Reserved Price!')

    def optimal_r_experimental(self):
        pass 


    def main(self, iterations, rp):
        # self.PlotGraph()
        total_rev = 0 
        run_rev = 0
        # rp = 0.6
        for _ in tqdm(range(iterations)):
            vs = self.ValGeneration('uniform', 0, 1, self.cnt - 1)
            # total_rev += self.IDMRev_RP(vs, rp)
            _, _, rev = self.IDM_with_rp(vs, rp)
            run_rev += rev 
        # return total_rev / iterations, run_rev / iterations
        # return total_rev / iterations
        return run_rev / iterations
    
    def test_rp(self, rps, iters):
        res = []
        # test different optimal revenues for different rps 
        for rp in rps:
            res.append([rp, self.main(iters, rp)])
        return res 
    

def plot_func(x, y_lists):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(16, 10))
    colors = ['red', 'yellow', 'blue', 'green', \
              'purple', 'orange', 'cyan', 'grey',\
                'lightblue', 'pink', 'tan']
    # y_lists中包含四幅图 
    y_lists_rps, y_lists_rev = [], []
    for y in y_lists:
        y_lists_rps.append([x[0] for x in y])
        y_lists_rev.append([x[1] for x in y])
    # print(y_lists_rev)
    y_lists_smooth = []
    for y in y_lists_rev:
        tmp = gaussian_filter1d(y, sigma = 10)
        y_lists_smooth.append(tmp)
    print('here', y_lists_smooth)
    if len(colors) < len(y_lists_smooth):
        return 'Colors not enough!'
    n = len(y_lists_smooth)
    legends = ['Tree-' + str(i+1) for i in range(n)]
    selected_colors = colors[:n]
    for y, c in zip(y_lists_smooth, selected_colors):
        print('debug', x, y)
        plt.plot(x, y, color = c)
    # 找到最大的值的位置并将点标出来
    max_x, max_y = [], []
    for i, y in enumerate(y_lists_smooth):
        idx = np.argmax(y)
        max_x.append(y_lists_rps[i][idx])
        max_y.append(max(y))
    # plt.scatter(max_x, max_y, marker = 'd', markersize = 20, color = 'black')
    for dx, dy in zip(max_x, max_y):
        plt.plot([dx], [dy], marker='d', markersize=20, color='black')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('reserved price', fontsize=18)
    plt.ylabel('revenue', fontsize=18)
    # plt.legend(('T0', 'T1', 'T2', 'T3', 'T4'))
    plt.legend(legends, fontsize=18)
    plt.tight_layout()
    plt.savefig('rooted_trees_with_4_bidders_smooth.pdf')
    plt.show()

def trees_with_three_bidders(n, edges_sets):
    nodes = [0,1,2,3]
    G_list = []
    for i in range(n):
        tmp_G = nx.Graph()
        tmp_G.add_nodes_from(nodes)
        tmp_G.add_edges_from(edges_sets[i])
        G_list.append(tmp_G)
    # G0.add_edges_from()
    return G_list 

def trees_with_four_bidders(n, edges_sets):
    nodes = [0,1,2,3,4]
    G_list = []
    for i in range(n):
        tmp_G = nx.Graph()
        tmp_G.add_nodes_from(nodes)
        tmp_G.add_edges_from(edges_sets[i])
        G_list.append(tmp_G)
    # G0.add_edges_from()
    return G_list


def calc_results(Graph_list, seller_id, buyer_ids, x_vals, iters):
    y_lists = []
    for G in Graph_list:
        tmp_idm = IDMVerify(G, seller_id, buyer_ids)
        tmp_res = tmp_idm.test_rp(x_vals, iters)
        # plot_res = [x[1] for x in tmp_res]
        y_lists.append(tmp_res)
    return y_lists

if __name__ == "__main__":
    seller_id = 0
    # buyer_ids = [1,2,3]
    buyer_ids = [1,2,3,4]
    # edges_sets = [
    #     [(0,1), (1,2), (2,3)],
    #     [(0,1), (1,2), (1,3)],
    #     [(0,1), (0,2), (1,3)], 
    #     [(0,1), (0,2), (0,3)]
    # ]
    edges_sets = [
        [(0,1), (1,2), (2,3), (3,4)],
        [(0,1), (1,2), (2,3), (2,4)],
        [(0,1), (1,2), (1,3), (2,4)],
        [(0,1), (1,2), (1,3), (1,4)],
        [(0,1), (0,2), (1,3), (1,4)],
        [(0,1), (0,2), (1,3), (2,4)],
        [(0,1), (0,2), (1,3), (3,4)],
        [(0,1), (0,2), (0,3), (1,4)],
        [(0,1), (0,2), (0,3), (0,4)]
    ]
    iters = 10000 
    # x_ranges = [0.01 * i for i in range(40, 81)]
    x_ranges = [0.002 * i for i in range(200, 401)]
    G_list = trees_with_four_bidders(9, edges_sets)
    y_res = calc_results(G_list, seller_id, buyer_ids, x_ranges, iters)
    plot_func(x_ranges, y_res)