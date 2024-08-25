class NeighborSampler(BlockSampler):
    """
    Neighbor sampling for multilayer GNN.
    """

    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        if self.fused and get_num_threads() > 1:
            cpu = F.device_type(g.device) == "cpu"
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu and F.device_type(seed_nodes[ntype].device) == "cpu"
                    )
            else:
                cpu = cpu and F.device_type(seed_nodes.device) == "cpu"
            if cpu and isinstance(g, DGLGraph) and F.backend_name == "pytorch":
                if self.g != g:
                    self.mapping = {}
                    self.g = g
                for fanout in reversed(self.fanouts):
                    block = g.sample_neighbors_fused(
                        seed_nodes,
                        fanout,
                        edge_dir=self.edge_dir,
                        prob=self.prob,
                        replace=self.replace,
                        exclude_edges=exclude_eids,
                        mapping=self.mapping,
                    )
                    seed_nodes = block.srcdata[NID]
                    blocks.insert(0, block)
                return seed_nodes, output_nodes, blocks

        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            block = to_block(frontier, seed_nodes)
            # If sampled from graphbolt-backed DistGraph, `EID` may not be in
            # the block.
            if EID in frontier.edata.keys():
                block.edata[EID] = frontier.edata[EID]
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
                
class BlockSampler(Sampler):
    """
    """

    def __init__(
        self,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__()
        self.prefetch_node_feats = prefetch_node_feats or []
        self.prefetch_labels = prefetch_labels or []
        self.prefetch_edge_feats = prefetch_edge_feats or []
        self.output_device = output_device


    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """Generates a list of blocks from the given seed nodes.
        返回triplet: input_nodes的id; output_nodes(最后一层节点)的id; 封装子图的block结构 
        """
        raise NotImplementedError

    def assign_lazy_features(self, result):
        """Assign lazy features for prefetching."""
        input_nodes, output_nodes, blocks = result
        set_src_lazy_features(blocks[0], self.prefetch_node_feats)
        set_dst_lazy_features(blocks[-1], self.prefetch_labels)
        for block in blocks:
            set_edge_lazy_features(block, self.prefetch_edge_feats)
        return input_nodes, output_nodes, blocks

    def sample(
        self, g, seed_nodes, exclude_eids=None
    ):  # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return self.assign_lazy_features(result)



def _find_exclude_eids_with_reverse_id(g, eids, reverse_eid_map):
    if isinstance(eids, Mapping):
        eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        exclude_eids = {
            k: F.cat([v, F.gather_row(reverse_eid_map[k], v)], 0)
            for k, v in eids.items()
        }
    else:
        exclude_eids = F.cat([eids, F.gather_row(reverse_eid_map, eids)], 0)
    return exclude_eids


def _find_exclude_eids_with_reverse_types(g, eids, reverse_etype_map):
    exclude_eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
    reverse_etype_map = {
        g.to_canonical_etype(k): g.to_canonical_etype(v)
        for k, v in reverse_etype_map.items()
    }
    for k, v in reverse_etype_map.items():
        if k in exclude_eids:
            if v in exclude_eids:
                exclude_eids[v] = F.unique(
                    F.cat((exclude_eids[k], exclude_eids[v]), dim=0)
                )
            else:
                exclude_eids[v] = exclude_eids[k]
    return exclude_eids


def _find_exclude_eids(g, exclude_mode, eids, **kwargs):
    if exclude_mode is None:
        return None
    elif callable(exclude_mode):
        return exclude_mode(eids)
    elif F.is_tensor(exclude_mode) or (
        isinstance(exclude_mode, Mapping)
        and all(F.is_tensor(v) for v in exclude_mode.values())
    ):
        return exclude_mode
    elif exclude_mode == "self":
        return eids
    elif exclude_mode == "reverse_id":
        return _find_exclude_eids_with_reverse_id(
            g, eids, kwargs["reverse_eid_map"]
        )
    elif exclude_mode == "reverse_types":
        return _find_exclude_eids_with_reverse_types(
            g, eids, kwargs["reverse_etype_map"]
        )
    else:
        raise ValueError("unsupported mode {}".format(exclude_mode))


def find_exclude_eids(
    g,
    seed_edges,
    exclude,
    reverse_eids=None,
    reverse_etypes=None,
    output_device=None,
):
    """
    """
    exclude_eids = _find_exclude_eids(
        g,
        exclude,
        seed_edges,
        reverse_eid_map=reverse_eids,
        reverse_etype_map=reverse_etypes,
    )
    if exclude_eids is not None and output_device is not None:
        exclude_eids = recursive_apply(
            exclude_eids, lambda x: F.copy_to(x, output_device)
        )
    return exclude_eids
