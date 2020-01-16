#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   corpus.py
@Time    :   2020/01/06 12:01:44
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from lxml import etree
from pathlib import Path

class Item:
    
    part_color_marker = zip(
        ["introduction", "section", "conclusion"],
        ["green", "brown", "blue"],
        [".", "x", "+"],
    )

    def __init__(self, abstract, 
                introduction, section, conclusion):
        self.abstract = abstract
        self.introduction = introduction
        self.section = section
        self.conclusion = conclusion

    @property
    def description(self):
        pass



class Corpus:
    # introduction 0.3, section 0.9, conclusion 1.0
    PORTION = [0.3, 0.9]

    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def items_generator(self, limit_size=200):
        cnt = 0
        for item in map(self.build_item, self._files):
            if not isinstance(item, Item): continue
            yield item
            if cnt == limit_size: break
            cnt += 1

    def split_sentences(self, sentences, points=PORTION):
        if len(points) <= 0: return sentences
        points.extend([0, 1])

        points = sorted(set(points))

        sentence_group = []
        sentence_len = len(sentences)
        for pre_point, after_point in zip(points[:-1], points[1:]):
            pre_point, after_point = map(int, 
                [sentence_len * pre_point, sentence_len * after_point])
            sentence_group.append(sentences[pre_point:after_point])

        return map(lambda lines: "".join(lines), sentence_group)

    def plot(self, mode="scatter", limit_size=200, show=True):
        description = [item.description 
            for item in self.item_generator(limit_size)]

        fig = plt.figure(figsize=(10, 5))
        for i, (title, part) in enumerate(
            ["proportion", "uniq_proportion"], 1):
            axis = fig.add_subplot(1, 2, i)
            axis.set_title(title)
            axis.legend()

            for part_name, color, marker in Item.part_color_marker:
                if mode == "scatter":
                    axis.scatter(range(len(description)), 
                        getattr(description, part_name), 
                        label=part_name, 
                        color=color, 
                        marker=marker)
                elif mode == "distribution":
                    sns.distplot(getattr(description, part_name),
                        label=part_name, color=color, ax=axis,
                        rug=False, hist=False)
        
        if show: 
            plt.show()

        return fig


class CNN(Corpus):

    @property
    def _files(self):
        return (self.base_path / "stories").glob("*.story")

    def _split_story(self, file):
        content = file.read_text()
        idx = content.find("@highlight")
        story, hightlights = content[:idx], content[idx:].split("@highlight")
        not_null_f = lambda line: len(line) > 0
        
        return (filter(not_null_f, story.split("\n")), 
            filter(not_null_f, hightlights))

    def build_item(self, file):
        def _pre_lines(lines):
            return [line[9:] if line.startswith("(CNN) -- ") else line
                for line in lines ]

        story_lines, hightlights = map(_pre_lines, self._split_story(file))

        return Item("".join(hightlights), *self.split_sentences(story_lines))


class Legal(Corpus):

    @property
    def _files(self):
        file_names = map(lambda path: path.name, 
                        (self.base_path / "citations_summ").glob("*.xml"))

        return map(lambda name: (self.base_path / "citations_summ" / name, 
                                self.base_path / "fulltext" / name),
                    file_names)

    def build_item(self, files):
        summ_tree, full_tree = map(lambda file: etree.HTML(file.read_text()), files)

        summ_phrases = ". ".join(summ_tree.xpath("//citphrase//text()"))
        cite_phrases = ". ".join(full_tree.xpath("//catchphrase//text()"))
        sentences = full_tree.xpath("//sentence//text()")[:-2]  # Irrelevant sentence to text

        return Item(f"{summ_phrases} {cite_phrases}", *self.split_sentences(sentences))


class SciSumm(Corpus):

    @property
    def _files(self):
        return map(lambda path: list((path / "Reference_XML").glob("*.xml"))[0],
            (self.base_path / "data/Training-Set-2019/Task2/From-ScisummNet-2019").iterdir())
    
    def build_item(self, file):
        tree = etree.HTML(file.read_text())

        abstract = "".join(tree.xpath("//abstract/s/text()"))
        if not abstract: 
            return "No <abstract>"

        sections = [( section.xpath("@title")[0], "".join(section.xpath("s/text()")) ) 
            for section in tree.xpath("//section")]
        sections = list(filter(lambda sec: sec[0] != "" and
           not re.search("(acknowledge?ment|reference|appendix|abstract)", sec[0].lower()), sections))

        if len(sections) < 3: 
            return "It needs more sections."

        introduction = section_content = conclusion = ""
        for title, content in sections:
            if re.search("(introduction|motivation|background)", title.lower()):
                introduction += content
            elif re.search("(conclust?ion|discussion|result|summary)", title.lower()):
                conclusion += content
            else:
                section_content += content
        
        if not introduction or not conclusion:
            return "Can't Find Introduction or Conclusion!"

        return Item(abstract, introduction, section_content, conclusion)



class ACL2014(Corpus):

    @property
    def _files(self):
        file_names = map(lambda path: path.name,
            (self.base_path / "abstract").glob("*.txt"))
        
        return map(lambda name: (self.base_path / "abstract" / name, 
                        self.base_path / "content" / name), 
            file_names)

    def build_item(self, files):
        abstract, full = map(lambda file: file.read_text(), files)
        title, introduction, *sections, conclusion = full.split("\n")

        return Item(abstract, introduction, " ".join(sections), conclusion)