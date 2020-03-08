#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sln.py
@Time    :   2020/01/16 12:42:37
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from collections import namedtuple

(
    SELF_LINK = Link("SELF"),
    TEMPORAL_LINK = Link("Temporal"),
    CAUSE_EFFECT_LINK = Link("Cause-Effect"),
    PURPORSE_LINK = Link("Purpose"),
    MEANS_LINK = Link("Means"),
    CONDITION_LINK = Link("Condition"),
    SEQUENTIAL_LINK = Link("Sequential"),
    ATTRIBUTION_LINK = Link("Attribution"),
)

Link = namedtuple("Link", ["node1", "node2", "type", "count"])

def fusion_SLN(sln1, sln2):
    links = []
    for sln_link in sln1.links + sln2.links:
        for link in links:
            if sln_link == link:
                links.remove(link)
                links.append(link._replace(count=link.count + sln_link))
                break
        else:
            links.append(sln_link)
    
    return SLN(links)


def framnet_sln_builder(tokens):
    links = []

    # Keep the important element    
    for token, tag in nltk.pos_tag(tokens):
        pass

    return SLN(links)

class SLN:
    """An environment to handle an SLN.
    """
    def __init__(self, links):
        self.links = links

    @property
    def nodes(self):
        _nodes = []
        for link in links:
            _nodes.extend([link.node1, link.node2])
        return sorted(set(_nodes))

    def update(self, node):
        self.nodes.append(node)

    def build_link(self, node1, node2):
        node1.build_SLN() node2.build_SLN()

    def build_links(self):
        for i, node_i in self.nodes:
            for j, node_j in self.nodes:
                if i == j: 
                    self.links[i][i] = SELF_LINK
                elif i > j :
                    self.links[i][j] = self.links[j][i]
                else:
                    self.links[i][j] = self.build_link(node_i, node_j)
    
    def summarize(self, token_importance):
        pass


class Node:
    """Node in SLN.

    Node can be a entity, a concept or event an SLN.
    """
    def __init__(self, tokens):
        self.tokens = tokens


class Link:
    """Semantic link in SLN.

    Link handles the properties and relationship between nodes.
    """
