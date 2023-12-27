document$.subscribe(({ body }) => {
    renderMathInElement(body, {
        delimiters: [
            {left: "$$", right: "$$", display: true},
            {left: "$", right: "$", display: false},
            {left: "\\(", right: "\\)", display: false},
            {left: "\\[", right: "\\]", display: true},
        ],
        throwOnError: false,
        macros: {
            // Statistical operators
            "\\E": "\\mathop{\\mathbb{E}}",
            "\\Var": "\\mathop{\\rm{Var}}",
            "\\Std": "\\mathop{\\rm{Std}}",
            "\\Cov": "\\mathop{\\rm{Cov}}",

            // Number sets
            "\\naturals": "\\mathbb{N}",
            "\\integers": "\\mathbb{Z}",
            "\\rationals": "\\mathbb{Q}",
            "\\reals": "\\mathbb{R}",
            "\\complexes": "\\mathbb{C}",

            // Beta function
            "\\B": "\\mathop{\\mathrm{B}}",

            // Tsallis' q-log and q-exp
            "\\qexp": "e_{#1}^{#2}",
            "\\qlog": "\\ln_{#1}\\left({#2}\\right)",

            // Falling and rising factorials
            "\\ffact": "\\left( #1 \\right)^{-}_{#2}",
            "\\rfact": "\\left( #1 \\right)^{+}_{#2}",

            // L-moments
            "\\lmoment": "\\lambda_{#1}",
            "\\tlmoment": "\\lambda^{(#1)}_{#2}",

            // L-moment ratio's
            "\\lratio": "\\tau_{#1}",
            "\\tlratio": "\\tau^{(#1)}_{#2}",

            // (shifted) Jacobi polynomial
            "\\jacobi": "P_{#1}^{(#2, #3)}\\left(#4\\right)",
            "\\shjacobi": "\\widetilde{P}_{#1}^{(#2, #3)}\\left(#4\\right)",
        }
    })
})
