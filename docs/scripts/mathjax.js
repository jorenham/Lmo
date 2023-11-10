window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        macros: {
            // Expectation operator
            E: "\\mathop{\\mathbb{E}}",
            // Beta function
            B: "\\mathop{\\mathrm{B}}",
            // Box-Cox transformation
            boxcox: ["\\rm \\mathcal{T}_{#2}\\left( #1 \\right)", 2],
            // L-moment (untrimmed)
            lmoment: ["\\lambda_{#1}", 1],
            // L-moment ratio (untrimmed)
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
