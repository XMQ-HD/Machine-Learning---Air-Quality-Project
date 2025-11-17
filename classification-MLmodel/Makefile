# LaTeX Report Compilation Makefile

# 主文件
MAIN = report
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

# LaTeX 编译器
LATEX = pdflatex
BIBTEX = bibtex

# 编译选项
LATEX_OPTS = -interaction=nonstopmode -file-line-error

# 默认目标
all: $(PDF)

# 编译 PDF
$(PDF): $(TEX)
	$(LATEX) $(LATEX_OPTS) $(TEX)
	$(LATEX) $(LATEX_OPTS) $(TEX)  # 第二次编译以解决引用

# 清理临时文件
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls

# 完全清理（包括 PDF）
cleanall: clean
	rm -f $(PDF)

# 查看 PDF（需要系统支持）
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "请手动打开 $(PDF)"; \
	fi

.PHONY: all clean cleanall view






