\documentclass[11pt]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{color}

% Page Layout
\geometry{
    a4paper,
    left=25mm,
    right=25mm,
    top=25mm,
    bottom=25mm,
}

% Hyperlink Colors
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

% Listings Configuration for Code
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{bashstyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{blue},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

% Title Configuration
\title{\textbf{Capsule Network App}}
\date{}

\begin{document}

% Title
\maketitle

% Sections
\section*{📚 Overview}
The \textbf{Capsule Network App} is a user-friendly tool designed to help users understand, train, test, and perform inference with Capsule Networks, a cutting-edge deep learning architecture introduced by Geoffrey Hinton. 

\section*{🚀 Features}
\begin{itemize}
    \item Interactive web-based UI built with \textbf{Streamlit}.
    \item Options to \textbf{Train}, \textbf{Test}, and \textbf{Infer} Capsule Network models.
    \item Detailed explanations and visual aids for better understanding of Capsule Networks.
    \item Tools to upload images for inference and customize training parameters.
\end{itemize}

\section*{🖥️ Directory Structure}
\begin{verbatim}
Capsule-Network-App/
├── app.py
├── images/
│   ├── architecture_capsnet.png
│   ├── working_capsnet.png
│   └── header_image.png
├── requirements.txt
└── README.md
\end{verbatim}

\begin{itemize}
    \item \textbf{app.py}: Main application file.
    \item \textbf{images/}: Folder containing visual aids and diagrams.
    \item \textbf{requirements.txt}: Python dependencies for the app.
    \item \textbf{README.md}: Instructions and details about the project.
\end{itemize}

\section*{⚙️ Installation}
\subsection*{Prerequisites}
\begin{itemize}
    \item Python 3.7 or higher (\href{https://www.python.org/downloads/}{Download here}).
    \item Git (\href{https://git-scm.com/downloads}{Download here}).
\end{itemize}

\subsection*{Installation Steps}
\begin{enumerate}
    \item Clone the repository:
    \begin{lstlisting}[style=bashstyle]
    git clone https://github.com/yourusername/Capsule-Network-App.git
    \end{lstlisting}
    \item Navigate to the project directory:
    \begin{lstlisting}[style=bashstyle]
    cd Capsule-Network-App
    \end{lstlisting}
    \item (Optional) Create a virtual environment:
    \begin{lstlisting}[style=bashstyle]
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    \end{lstlisting}
    \item Install dependencies:
    \begin{lstlisting}[style=bashstyle]
    pip install -r requirements.txt
    \end{lstlisting}
    \item Run the application:
    \begin{lstlisting}[style=bashstyle]
    streamlit run app.py
    \end{lstlisting}
\end{enumerate}

\section*{🛠️ Training, Testing, and Inference}
\subsection*{🏋️ Train}
Customize training parameters such as batch size, epochs, and learning rate, then click \texttt{Start Training} to begin.

\subsection*{🧪 Test}
Load a trained model, specify the test dataset, and evaluate metrics like accuracy and F1-score.

\subsection*{🔍 Infer}
Upload an image to the app, and it will preprocess and display the model's predictions.

\section*{🤝 Contributing}
Contributions are welcome! Please follow the steps:
\begin{enumerate}
    \item Fork the repository.
    \item Create a new branch:
    \begin{lstlisting}[style=bashstyle]
    git checkout -b feature/YourFeatureName
    \end{lstlisting}
    \item Commit changes:
    \begin{lstlisting}[style=bashstyle]
    git commit -m "Add YourFeatureName"
    \end{lstlisting}
    \item Push to your branch:
    \begin{lstlisting}[style=bashstyle]
    git push origin feature/YourFeatureName
    \end{lstlisting}
    \item Open a pull request.
\end{enumerate}

\section*{📄 License}
Licensed under the \href{https://opensource.org/licenses/MIT}{MIT License}.

\section*{🙏 Acknowledgments}
\begin{itemize}
    \item \textbf{Geoffrey Hinton} for Capsule Networks.
    \item \textbf{Streamlit} for the web application framework.
    \item \textbf{OpenAI} for AI research and resources.
\end{itemize}

\hrule
\begin{center}
    \textit{© 2024 Capsule Network App. All rights reserved.}
\end{center}

\end{document}
