window.MathJax = {
    loader: {
        load: ["[tex]/braket"]
    },
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"], ["$$", "$$"]],
        processEscapes: true,
        processEnvironments: true,
        tags: "ams",
        packages: {
            "[+]": ["braket"]
        },
        macros: {
            // Expectation operator
            E: "\\mathop{\\mathbb{E}}",
            // Beta function
            B: "\\mathop{\\mathrm{B}}",
            // Tsallis' q-log and q-exp
            qexp: ["e_{#1}^{#2}", 2],
            qlog: ["\\ln_{#1}\\left({#2}\\right)", 2],
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
            tlratio: ["\\tau^{(#1)}_{#2}", 2],
            // Jacobi polynomial
            jacobi: ["P_{#1}^{(#2, #3)}\\left(#4\\right)", 4],
            // Shifted Jacobi polynomial
            shjacobi: ["\\widetilde{P}_{#1}^{(#2, #3)}\\left(#4\\right)", 4]
        }
    },
    options: {
        processHtmlClass: "arithmatex"
    },
    chtml: {
        displayAlign: "left"
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
