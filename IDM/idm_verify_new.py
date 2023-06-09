import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import random 
import types
import math 
import collections 
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d 
from scipy import optimize

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
        dominating_dict[principal] = set()
        for node in agent_set: 
            flag = True 
            for agent in agent_set:
                if node in dominating_dict[agent]:
                    flag = False 
                    break 
            if flag:
                dominating_dict[principal].add(node)
        print(dominating_dict)
        return dominating_dict

    # def _Construct_DS(self) -> dict:


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
        seller_dominates = dominating_dict.pop(0)
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
    
    def Calculate_Opt_rp(self):
        '''
        求解最优保留价逻辑
        sum_i=1^m r^{-alpha_i} = n + m 
        考虑只设置一个最优的保留价
        计算每个分支的数量: alpha1, alpha2, ..., alpham 
        n -> 市场中所有竞拍者的数量
        m -> m个分支的数量
        '''
        dominant_set = self._ConstructDS()
        print(dominant_set)
        seller_dominant = dominant_set[self.seller]
        alphas = []
        n = len(self.buyers)
        m = len(seller_dominant)
        for buyer in seller_dominant:
            alphas.append(len(dominant_set[buyer]) + 1)
        # define the equation that needs to be solved
        # print(alphas) 
        def func(x):
            y = 0 
            for alpha in alphas:
                y += (1/x) ** alpha
            return y - n - m 
        # use the solve function from scipy 
        print(n, m, alphas)
        root = optimize.root(func, 0.1)
        if not root:
            raise ValueError("No Proper Reserved Price!")
        return root['x'][0]

    # IDM带保留价结果
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
        seller_dominates = dominating_dict.pop(0)
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

    '''
    目标是对比三种不同的保留价模式下的收益情况分析
    给定一张确定的图 -> 唯一的保留价就是确定的 -> 按分支确定的保留价也是确定的
    1. 将所有竞拍者的保留价映射到经典的虚拟估值函数上进行
    2. 按照分支给出最优的保留价然后执行机制
    3. 按照分支上的虚拟估值进行处理 同时使用truthful的框架来保证出价与传播的IC性质
    4. 一般IDM执行后的收益情况

    '''
    # IDM带经典Myerson虚拟估值场景
    def IDM_with_classical_virtual_value(self, vals):
        '''
        这里的逻辑是将每个竞拍者的估值映射到经典的虚拟估值空间上
        这里以[0,1]区间的均匀分布为例子 虚拟估值的函数映射到 2v - 1
        这种的弊端是没有考虑到网络结构信息, 相当于设置了一个确定的保留价进行IDM 
        输入: 出价向量
        输出: 分配以及对应的支付
        '''
        def affine_func(x):
            return 2 * x - 1
        def reverse_affine_func(fx):
            return (fx + 1) / 2 
            
        virtual_vals = [affine_func(v) for v in vals] # 所有竞拍者虚拟估值的结果
        # 基于虚拟估值找到社会福利最大化的分配
        # 找到虚拟估值最大的竞拍者
        # 如果最大的虚拟估值都是大于0的直接不分配 
        # print(virtual_vals)
        # virtual_profile = [[buyer_id, bid] for buyer_id, bid in zip(self.buyers, virtual_vals)]
        if not virtual_vals:
            raise ValueError("No Input Virtual Vals!")
        if max(virtual_vals) < 0:
            return None, None, 0
        # there exists some bidder whose virtual_valuation is higher than zero 
        # run IDM under the virtual value space and return the corresponding payment  
        winner, rewarded_bidders, _ = self.IDM(virtual_vals)
        # reflect the virtual payment into the real payment under initial value space 
        print('here', winner, rewarded_bidders)
        for bidder in winner:
            winner[bidder] = reverse_affine_func(winner[bidder])
        for r_bidder in rewarded_bidders:
            rewarded_bidders[r_bidder] = reverse_affine_func(rewarded_bidders[r_bidder])
        rev = sum(winner.values()) + sum(rewarded_bidders.values())
        return winner, rewarded_bidders, rev

    # IDM带每个分支最优保留价结果
    def IDM_with_branch_reserved_price(self, vals):
        branch_rp = dict()
        dominate_set = self._ConstructDS()
        keys = dominate_set[self.seller]
        for ky in keys:
            d_ky = len(dominate_set[ky])
            rp_ky = 1 / math.pow((1+d_ky), 1/d_ky)
            branch_rp[ky] = rp_ky # 每个分支给一个保留价
        # 也可以写成对于每个竞拍者进行对应的价格歧视
        # 理论上这里的branch_rp只会用到一个
        # locate the highest bid
        d = {k:v for k, v in zip(self.buyers, vals)} 
        m = list(d.items())
        m.sort(key = lambda x : -x[1])
        if not m:
            raise ValueError('no buyer exists!')
        highest_bidder = m[0][0]
        used_rp = None 
        for ky in keys:
            if highest_bidder in dominate_set[ky]:
                used_rp = branch_rp[ky]
                break 
        if not used_rp:
            raise ValueError("No Reserved Price!")
        return self.IDM_with_rp(vals, used_rp)

    def IDM_with_branch_virtual_value(self, vals):
        '''
        每一个支配的分支定义新的虚拟估值 价格歧视落在不同分支上
        '''
        def new_affine_func(x, k):
            return x - ((1 - x ** k) / k * (x ** (k-1)))

        virtual_vals = dict()
        d = {k:v for k, v in zip(self.buyers, vals)}
        dominant_set = self._ConstructDS()
        first_level_critial_bidders = dominant_set[self.seller]
        

    
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
    # seller_id = 0
    # buyer_ids = [1,2,3]
    # buyer_ids = [1,2,3,4]
    # edges_sets = [
    #     [(0,1), (1,2), (2,3)],
    #     [(0,1), (1,2), (1,3)],
    #     [(0,1), (0,2), (1,3)], 
    #     [(0,1), (0,2), (0,3)]
    # ]
    # edges_sets = [
    #     [(0,1), (1,2), (2,3), (3,4)],
    #     [(0,1), (1,2), (2,3), (2,4)],
    #     [(0,1), (1,2), (1,3), (2,4)],
    #     [(0,1), (1,2), (1,3), (1,4)],
    #     [(0,1), (0,2), (1,3), (1,4)],
    #     [(0,1), (0,2), (1,3), (2,4)],
    #     [(0,1), (0,2), (1,3), (3,4)],
    #     [(0,1), (0,2), (0,3), (1,4)],
    #     [(0,1), (0,2), (0,3), (0,4)]
    # ]
    # iters = 10000 
    # # x_ranges = [0.01 * i for i in range(40, 81)]
    # x_ranges = [0.002 * i for i in range(200, 401)]
    # G_list = trees_with_four_bidders(9, edges_sets)
    # y_res = calc_results(G_list, seller_id, buyer_ids, x_ranges, iters)
    # plot_func(x_ranges, y_res)
    '''
    compare revenue under different graphic structured market 
    '''
    # G1 = nx.karate_club_graph()
    G1 = nx.Graph()
    G1.add_nodes_from([0,1,2,3,4])
    G1.add_edges_from([(0,1), (1,2), (0,3), (3,4)])
    print(G1.nodes())
    nx.draw(G1, with_labels=True)
    plt.show()
    seller_id = 0 
    bidder_number = len(G1.nodes())
    buyer_ids = [i for i in range(1, bidder_number)]
    test_idm = IDMVerify(G1, seller_id, buyer_ids)
    print(test_idm.Calculate_Opt_rp())
    vals = test_idm.ValGeneration('uniform',0, 1, bidder_number - 1)
    winner, rewarded_bidders, rev = test_idm.IDM(vals)
    opt_rp = test_idm.Calculate_Opt_rp()
    new_winner, new_rewarded_bidders, new_rev = test_idm.IDM_with_rp(vals, opt_rp)
    winner2, rewarded_bidders2, rev2 = test_idm.IDM_with_classical_virtual_value(vals)
    print(winner, rewarded_bidders, rev)
    print(new_winner, new_rewarded_bidders, new_rev)
    print(winner2, rewarded_bidders2, rev2)

