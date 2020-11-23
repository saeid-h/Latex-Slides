
#############################################################################
#																			#
# Copyright (c) 2020 Saeid Hosseinipoor <https://saeid-h.github.io/>		#
# All rights reserved.														#
# Licensed under the MIT License											#
#																			#
############################################################################# 

import os

class LatexObject(object):
    pass

class Graphics(LatexObject):
    def __init__(self, scale=1.0, path=None, frame=True):
        super(Graphics, self).__init__()
        self.scale = scale
        self.path = path
        self.frame = frame

    def build_latex(self):
        latex = ''
        if self.frame: latex += '\\frame{'
        latex += '\\includegraphics[scale={}]{{{}}}'.format(self.scale, self.path)
        if self.frame: latex += '}'
        self.latex = latex
        return latex

class PythonCode(LatexObject):
    def __init__(self, code=None):
        super(PythonCode, self).__init__()
        self.code = code

    def build_latex(self):
        latex = '\\begin{python}\n'
        latex += self.code + '\n'
        latex += '\\end{python}\n'
        self.latex = latex
        return latex

class Items(LatexObject):
    def __init__(self, content=None):
        self.content = content

    def build_latex(self):
        if self.content is None: return ''
        latex = '\\begin{itemize}\n'
        for item in self.content:
            if isinstance(item,LatexObject):
                latex += item.build_latex()
            else:
                latex += '\\item ' + str(item) + '\n'
        latex += '\\end{itemize}\n'
        self.latex = latex
        return latex


class Tables(LatexObject):
    def __init__(self, header=['Competitor' 'Name', 'Swim', 'Cycle', 'Run', 'Total'],
                        adjustments=['l']+['c']*4,
                        vertical_line=False,
                        horizontal_line=False,
                        data=[['John', '13:04', '24:15', '18:34', '55:53']],
                        caption=None):
        super(Tables, self).__init__()
        self.header = header
        self.data = data
        self.adjustments = adjustments
        self.caption = caption
        self.horizontal_line = horizontal_line
        self.vertical_line = vertical_line

    def add_data(self, new_data):
        if self.data is None:
            self.data = []
        self.data.append(new_data)
    
    def build_latex(self):
        latex = '\\begin{table}\n'
        latex += '\\begin{tabular}{'
        for adj in self.adjustments:
            latex += adj 
            if self.vertical_line: latex+= ' |'
        latex += '}\n'
        if self.header:
            latex_stack = []
            for item  in self.header:
                if isinstance(item,LatexObject):
                    latex_stack += [item.build_latex()]
                else:
                    latex_stack += [str(item)]
            latex += '& '.join(latex_stack) + '\\\\\n'
            if self.horizontal_line: latex += '\\hline \\hline\n'
        for line in self.data:
            latex_stack = []
            for item in line:
                if isinstance(item,LatexObject):
                    latex_stack += [item.build_latex()]
                else:
                    latex_stack += [str(item)]
            latex += '& '.join(latex_stack) + '\\\\\n'
        latex += '\\end{tabular}\n'
        if self.caption: latex += '\\caption{{{}}}\n'.format(self.caption)
        latex += '\\end{table}\n'
        self.latex = latex
        return latex


class Figures(LatexObject):
    def __init__(self,content=None,
                label=None,
                caption=None,
                text=None):
        super(Figures, self).__init__()
        self.label = label
        self.caption = caption
        self.text = text
        self.content = content

    def build_latex(self):
        latex = '\\begin{figure}\n'
        if isinstance(self.content,LatexObject):
            latex += self.content.build_latex()
        else:
            latex += str(self.content) + '\n'
        if self.caption: latex += '\\caption{{{}}}\n'.format(self.caption)
        if self.label: latex += '\\label{{{}}}\n'.format(self.label)
        latex += '\\end{figure}\n'
        if self.text: latex += self.text + '\n'
        self.latex = latex
        return latex



class Frame(LatexObject):
    def __init__(self, title='Sample frame title',
                        content=None):
        super(Frame, self).__init__()
        self.title = title
        self.content = content

class LatexDocument(LatexObject):
    def __init__(self, title='Sample title',
                        subtitle=None,
                        institute='Stevens Institute of Technology',
                        author='Saeid Hosseinipoor',
                        date='\\today', 
                        notes='look at the code @ github.com/saeid-h/latex_slides', 
                        more_setup = None):
        super(LatexDocument, self).__init__()
        self.comment = "%Information to be included in the title page:"
        self.title = title
        self.subtitle = subtitle
        self.institute = institute
        self.author = author
        self.date = date
        self.notes = notes
        self.more_setup = more_setup

class LatexSlideDocument(LatexDocument):
    def __init__(self, title='Sample title',
                        subtitle=None,
                        institute='Stevens Institute of Technology',
                        author='Saeid Hosseinipoor',
                        date='\\today', 
                        notes='look at the code @ github.com/saeid-h/latex_slides',
                        packages=['utopia'],
                        theme = 'Boadilla',
                        usecolortheme='default',
                        font_size=8,
                        more_setup = None,
                        TOC=False):
        super(LatexSlideDocument, self).__init__(title, subtitle, institute, author,date, notes, more_setup)
        self.latex = None
        self.packeges = ['inputenc', 'utopia']
        if packages:
            for package in packages:
                self.packeges.append(package)
        self.theme = theme
        self.usecolortheme = usecolortheme
        self.packages = packages
        self.TOC = TOC
        self.frames = []
        self.font_size = font_size

    def add_frames(self, frames):
        if isinstance(frames, Frame):
            self.frames.append(frames)
        elif isinstance(frames, list):
            for frame in frames:
                self.frames.append(frame)


    def __build_title_page(self):
        latex = '\\documentclass[{}pt]{{{}}}\n\n'.format(self.font_size, 'beamer')
        latex += '\\usepackage[utf8]{inputenc}\n'
        if self.packages:
            for package in self.packages:
                latex += '\\usepackage{{{}}}\n'.format(package)
        if self.more_setup: latex+= self.more_setup + '\n\n'
        latex += '\\usetheme{{{}}}\n'.format(self.theme)
        latex += '\\usecolortheme{{{}}}\n'.format(self.usecolortheme)
        latex += '\n\n'
        latex += self.comment + '\n'
        latex += '\\title{{{}}}\n'.format(self.title)
        if self.subtitle: latex += '\\subtitle{{{}}}\n'.format(self.subtitle)
        latex += '\\author{{{}}}\n'.format(self.author)
        latex += '\\institute{{{}}}\n'.format(self.institute)
        latex += '\\date{{{}}}\n\n\n\n\n'.format(self.date)
        # if self.notes: latex += '\\title{{{}}}\n'.format(self.notes)
        self.latex = latex
        return latex


    def __build_TOC(self, TOC_title='Table of Contents'):
        latex = '\\begin{frame}\n'
        latex += '\\frametitle{{{}}}\n'.format(TOC_title)
        latex += '\\tableofcontents\n'
        latex += '\\end{frame}\n\n'
        latex += '\\AtBeginSection[]\n'
        latex += '{\n \\begin{frame}\n'
        latex += '\\frametitle{{{}}}\n'.format(TOC_title)
        latex += '\\tableofcontents[currentsection]\n'
        latex += '\\end{frame}\n}\n\n'
        if self.latex is None: self.latex='' 
        self.latex += latex
        return latex


    def build_slides(self):
        latex = self.__build_title_page()   

        latex += '\\begin{document}\n\n'
        latex += '\\frame{\\titlepage}\n\n'

        if self.TOC: latex += self. __build_TOC() 

        for frame in self.frames:
            latex += '\\begin{frame}\n'
            latex += '\\frametitle{{{}}}\n'.format(frame.title)
            if isinstance(frame.content, LatexObject):
                latex += frame.content.build_latex()
            else:
                latex += str(frame.content) + '\n'
            latex += '\\end{frame}\n\n\n'
        latex += '\\end{document}\n'
        self.latex = latex
        return latex

