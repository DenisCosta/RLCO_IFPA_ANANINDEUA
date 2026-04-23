clear all; clc; close all;
fprintf('\n============================================================\n')
fprintf('\nInstituto Federal de Educação, Ciência e Tecnologia do Pará')
fprintf('\n                  IFPA Câmpus Ananindeua')
fprintf('\n  Grupo de pesquisa: Gradiente de Modelagem Matemática e')
fprintf('\n             Simulação Computacional - GM²SC')
fprintf('\n   Código-fonte utilizado nas simulações do RLCO Otimizador')
fprintf('\nArtigo publicado na Revista Científica Semana Acadêmica - Qualis A')
fprintf('\n  Autor: Dr. Denis C. L. Costa - Professor Titular do IFPA')
fprintf('\n       Doutorado e Pós-Doutorado em Sistemas de Energia\n')
fprintf('\n============================================================\n\n')
%% -----------------------------------------------------------------------------
% RLC Otimizador – Algoritmo de Metaheurísticas fundamentadas na análise
%                   Eletromagnética de Circuitos Elétricos
% Professor Dr. Denis Carlos Lima Costa - https://orcid.org/0000-0003-3207-6934
% Disponível em:
% https://semanaacademica.org.br/artigo/rlc-otimizador-algoritmo-de-
%         metaheuristicas-fundamentadas-na-analise-eletromagnetica-de
% Conceito: Ressonância com Ajuste Dinâmico e Chaves de Restrição
% O Ajuste Dinâmico (Auto-Adaptação): Código inteligente
%% -----------------------------------------------------------------------------

%% Configuração iniciais
% Tamanho da população (número de agentes de busca): nOsciladores
nOsciladores = 50;
maxIter = 150; % Número máximo de iterações
dim = 2;        % Quantidade de variáveis de controle
% Limites inferiores e superiores das variáveis de controle
lb = [-10, -10]; ub = [10, 10];

%% RLC OTIMIZADOR com Restrições
% Interruptor das Restrições: 1 => Ativo; 0 => Inativo
usar_h1 = 1; % Linear: x2 - 1.5*x1 = 0
usar_h2 = 1; % Não-Linear: x1^2 + x2^2 - 25 = 0
usar_g1 = 1; % Linear: x1 + x2 - 4 <= 0
usar_g2 = 1; % Não-Linear: x1^2 - x2 + 1 <= 0

% Parâmetros do Circuito Virtual
V_fonte = 100;  % Tensão da Fonte
L = 1.0;        % Indutância
C = 0.5;        % Capacitância
lambda = 100;   % Peso inicial para igualdades h(x)
mu = 100;       % Peso inicial para desigualdades g(x)

% Controle de Resistência Virtual: beta
beta = 1.1; % Fator de Incremento/Decremento das Penalidades.
% Comutador de porcentagem de osciladores: target_viability (entre 0 e 1)
target_viability = 0.3;

%% Inicialização
% Posição atual de cada oscilador no espaço de busca: omega
omega = lb + (ub - lb) .* rand(nOsciladores, dim);
% Atualização de Movimento (Aprendizado): omega é a variável que armazena
% onde está a posição da melhor solução (bestPos).
v = zeros(nOsciladores, dim);
bestPos = omega(1,:);
bestFit = inf;
history = zeros(maxIter, 1);

%% Loop de Otimização
% Fator de Qualidade: Q_factor
% Q_factor indica o quão "seletivo" ou "agudo" é o pico de ressonância.
for t = 1:maxIter
    Q_factor = 2 * (1 - (t/maxIter)^2);
    viables = 0;

    for i = 1:nOsciladores
        x1 = omega(i,1);
        x2 = omega(i,2);

        % Função Objetivo: f_val é tratada como a Resistência (R)
        f_val = x1^2 + x2^2 - 4*x1 - 6*x2 + 13;

        % Cálculo das Restrições e Violações
        v_h = 0; % Acumulador de violação de igualdade
        v_g = 0; % Acumulador de violação de desigualdade

        % Igualdade 1 (Linear)
        if usar_h1
            h1 = x2 - 1.5*x1;
            v_h = v_h + abs(h1);
        end

        % Igualdade 2 (Não-Linear: Círculo x1^2 + x2^2 = 25)
        if usar_h2
            h2 = x1^2 + x2^2 - 25;
            v_h = v_h + abs(h2);
        end

        % Desigualdade 1 (Linear)
        if usar_g1
            g1 = x1 + x2 - 4;
            v_g = v_g + max(0, g1);
        end

        % Desigualdade 2 (Não-Linear: Parábola x1^2 + 1 <= x2)
        if usar_g2
            g2 = x1^2 - x2 + 1;
            v_g = v_g + max(0, g2);
        end

        % Verificação de Viabilidade
        if v_h < 1e-3 && v_g <= 1e-3
            viables = viables + 1;
        end

        % Função de Penalidade (Impedância R)
        R = f_val + lambda * v_h^2 + mu * v_g^2;

        % Dinâmica do Circuito RLC
        X_L = sum(omega(i,:)) * L;
        X_C = 1 / (max(sum(omega(i,:)), 1e-3) * C + 1e-6);
        Z = sqrt(R^2 + (X_L - X_C)^2);
        I = V_fonte / (Z + 1e-6);

        % A fase phi: indica o atraso ou avanço entre a tensão e a corrente
        % devido aos componentes L e C.
        phi = atan((X_L - X_C) / (R + 1e-6));

        if R < bestFit
            bestFit = R;
            bestPos = omega(i,:);
        end

        % Movimento >> Se um oscilador está em uma posição omega com um valor
        % de função baixo (boa solução), a "Resistência" (R) diminui.
        r1 = rand(1, dim);

        % Controle do Tamanho do Passo (Amplitude)
        passo = I * cos(phi) * r1 .* (bestPos - omega(i,:)) * Q_factor;
        v(i,:) = 0.5 * v(i,:) + passo;
        omega(i,:) = omega(i,:) + v(i,:);
        omega(i,:) = max(min(omega(i,:), ub), lb);
    end

    % Ajuste Adaptativo das Penalidades: lambda e mu (coeficientes)
    % lambda e mu não são fixos: eles mudam conforme o desempenho da população
    v_rate = viables / nOsciladores;
    if v_rate < target_viability

        % lambda: Penaliza as violações de Igualdades (h)
        lambda = lambda * beta;

        % mu: Penaliza as violações de Desigualdades (g)
        mu = mu * beta;
    elseif v_rate > 0.8
        lambda = lambda / beta;
        mu = mu / beta;
    end
    history(t) = bestFit;
end

%% Representação Gráfica
fprintf('>>>>> Resultados RLCO Seletivo <<<<<\n');
fprintf('Status das Restrições: h1:%d, h2:%d, g1:%d, g2:%d\n', usar_h1,...
        usar_h2, usar_g1, usar_g2);
fprintf('X(1) otimizado: %.4f | X(2) otimizado: %.4f\n', bestPos(1),...
        bestPos(2));
fprintf(' Valor da Função : %.6f\n', bestFit);

% Visualização
figure('Color', 'k', 'Name', 'Análise de Convergência e Restrições',...
       'Position', [100, 100, 1100, 500]); 

% Convergência
subplot(1,2,1);
semilogy(history, 'LineWidth', 2,'Color', 'r');
title('Curva de Convergência'); grid on;
xlabel('Iterações'); ylabel('Z (Impedância/Custo)');

% Espaço de Busca e Restrições
subplot(1,2,2);
[X1, X2] = meshgrid(lb(1):0.1:ub(1), lb(2):0.1:ub(2));
F_plot = X1.^2 + X2.^2 - 4.*X1 - 6.*X2 + 13;
contour(X1, X2, F_plot, 30, 'LineWidth', 1, 'DisplayName', 'Função Objetivo');
hold on;

% h1: Linear Igualdade (Verde)
if usar_h1
    H1_plot = X2 - 1.5*X1;
    contour(X1, X2, H1_plot, [0 0], 'g', 'LineWidth', 2, 'DisplayName', 'h1: x2 - 1.5x1 = 0');
end

% h2: Não-Linear Igualdade (Vermelho)
if usar_h2
    H2_plot = X1.^2 + X2.^2 - 25;
    contour(X1, X2, H2_plot, [0 0], 'r', 'LineWidth', 2, 'DisplayName', 'h2: x1^2 + x2^2 = 25');
end

% g1: Linear Desigualdade (Ciano)
if usar_g1
    G1_plot = X1 + X2 - 4;
    contour(X1, X2, G1_plot, [0 0], 'c--', 'LineWidth', 2, 'DisplayName', 'g1: x1 + x2 <= 4');
end

% g2: Não-Linear Desigualdade (Magenta)
if usar_g2
    G2_plot = X1.^2 - X2 + 1;
    contour(X1, X2, G2_plot, [0 0], 'm--', 'LineWidth', 2, 'DisplayName', 'g2: x1^2 - x2 + 1 <= 0');
end

% Exibição do Ótimo Encontrado
plot(bestPos(1), bestPos(2), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y',...
     'DisplayName', 'Ótimo Encontrado');
title('Mapa de Busca e Fronteiras');
xlabel('x1'); ylabel('x2');
legend('Location', 'southeast'); % Outras posições da legenda: 'north','south',...
% 'east','west','northeast','northwest','southwest','southeast' ou 'best'. 
colorbar; colormap('jet'); grid on;
%% -----------------------------------------------------------------------------

