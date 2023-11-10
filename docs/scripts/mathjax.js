window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        macros: {
            boxcox: ["\\rm \\mathcal{T}_{#2}(#1)", 2],
            lmoment: ["\\lambda_{#1}", 1],
            lratio: ["\\tau_{#1}", 1]
        }
    },
    options: {
        // ignoreHtmlClass: ".*",
        processHtmlClass: "arithmatex"
    },
    chtml: {
        displayAlign: "left"
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
