window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        tags: "ams",
        macros: {
            // Expectation operator
            E: "\\mathop{\\mathbb{E}}",
            // Beta function
            B: "\\mathop{\\mathrm{B}}",
            // Box-Cox transformation, and its inverse
            boxcox: ["\\rm \\psi_{#2}\\left( #1 \\right)", 2],
            coxbox: ["\\rm \\psi^{-1}_{#2}\\left( #1 \\right)", 2],
            // Falling and rising factorials
            ffact: ["\\left( #1 \\right)^{-}_{#2}", 2],
            rfact: ["\\left( #1 \\right)^{+}_{#2}", 2],
            // L-moment (untrimmed)
            lmoment: ["\\lambda_{#1}", 1],
            // L-moment (trimmed)
            tlmoment: ["\\lambda^{(#1)}_{#2}", 2],
            // L-moment ratio (untrimmed)
            lratio: ["\\tau_{#1}", 1],
            // L-moment ratio (trimmed)
            tlratio: ["\\tau^{(#1)}_{#2}", 2]
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
