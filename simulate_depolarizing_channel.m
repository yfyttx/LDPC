function simulate_depolarizing_channel()
    % --- 1. 配置 ---
    
    % 添加译码器路径
    addpath('NB_LDPC/BP-decoder-for-NB_LDPC-codes-master');
    
    % 使用 C++ 构造器生成的 Alist 文件
    alist_file = 'my_generated_code.alist'; 
    
    % Alist 文件的参数 (来自 H_gamma.txt)
    q = 16;                  % 伽罗瓦域 GF(16)
    p = log2(q);             % p = 4
    
    % 仿真参数
    max_iter = 50;           % 译码最大迭代
    
    % X 轴：翻转概率 f_m (与论文图4一致)
    fm_list = 0.01:0.005:0.06; 
    
    max_frames = 10000;      % 每个 f_m 点仿真的最大帧数 (提高此值以获得更平滑的曲线)
    min_err_frames = 100;    % 至少统计 100 个错误帧 (论文中是 5x10^-5，需要极大仿真量)

    % --- 2. 初始化译码器和参数 ---
    
    % 初始化解码器 (MEX)
    [decoder_handle, N, M] = decode_soft(0, alist_file);
    
    % 码率计算
    % 经典码率 R_C = 1 - M/N。
    % 论文中的量子码率 R_Q = 1 - 2*J/L。对于 J=2, L=6，R_Q = 1/3。
    % 仿真中我们只需要 N 和 p。
    R_classic = max(1 - M / N, 1e-3);
    fprintf('开始去极化信道仿真 (论文图4复现):\n');
    fprintf('GF(%d), p=%d, N=%d, M=%d, R_c≈%.4f\n\n', q, p, N, M, R_classic);

    % 预先计算所有 q 个符号的汉明权重 (W_H)
    % symbols = (0:q-1)';
    % weights = sum(de2bi(symbols, p), 2);
    % 使用 bitget 更高效
    symbols = (0:q-1)';
    weights = sum(bitget(symbols, 1:p), 2);
    
    % 初始化结果数组
    BER = zeros(size(fm_list));
    FER = zeros(size(fm_list)); % FER 即论文中的 Block Error Rate

    % --- 3. 开始仿真循环 ---
    for idx = 1:numel(fm_list)
        fm = fm_list(idx);
        
        % 计算论文公式 (453) 的对数似然 (Log-Probability)
        % log(p(e)) = W_H(e)*log(fm) + (p - W_H(e))*log(1-fm)
        log_prob_fm = log(fm);
        log_prob_1_fm = log(1 - fm);
        
        % 这 p_n^(0) 是一个 qx1 向量，包含所有符号的初始对数概率
        % 
        initial_log_probs = weights * log_prob_fm + (p - weights) * log_prob_1_fm;
        
        % 假设所有 N 个变量节点具有相同的初始概率
        % (这是标准的全零码字仿真)
        % 'llrs' 矩阵是 qxN，是译码器的输入
        llrs = repmat(initial_log_probs, 1, N);
        
        total_bit_errors = 0;
        total_frame_errors = 0;
        total_frames = 0;
        
        fprintf('   - 正在仿真 f_m = %.4f\n', fm);
        
        while (total_frame_errors < min_err_frames) && (total_frames < max_frames)
            total_frames = total_frames + 1;
            
            % 1. 发送：假设全零码字 (符号 0)
            % 2. 信道：在译码端表现为 'llrs'
            
            % 3. 译码
            % [cite: 435-437]
            [ok, iters, hard_sym, ~] = decode_soft(2, decoder_handle, llrs, max_iter); %#ok<ASGLU>
            
            % 4. 错误统计
            % 我们发送的是全零 (符号 0)，所以任何非零符号都是错误
            sym_errs_in_frame = sum(hard_sym ~= 0);
            
            if sym_errs_in_frame > 0
                total_frame_errors = total_frame_errors + 1;
                
                % 计算此帧中的总比特错误数
                % (等于所有错误符号的汉明权重之和)
                error_symbols = hard_sym(hard_sym ~= 0);
                bit_errs_in_frame = sum(bitget(error_symbols, 1:p), 'all');
                total_bit_errors = total_bit_errors + bit_errs_in_frame;
            end
            
            if mod(total_frames, 1000) == 0
                 fprintf('      ...已完成 %d 帧, 发现 %d 帧错误。\n', total_frames, total_frame_errors);
            end
        end
        
        % 计算此 f_m 点的误码率
        num_bits_per_frame = N * p;
        BER(idx) = total_bit_errors / (total_frames * num_bits_per_frame);
        FER(idx) = total_frame_errors / total_frames;
        
        fprintf('   - 结果: BER = %.2e, FER (Block Error Rate) = %.2e\n\n', BER(idx), FER(idx));
    end
    
    fprintf('===== 仿真完成 =====\n\n');

    % --- 4. 绘图 (类似论文图4) ---
    fprintf('5. 正在绘制曲线图...\n');
    figure;
    semilogy(fm_list, FER, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    semilogy(fm_list, BER, 'bs--', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;
    xlabel('Flip Probability f_m');
    ylabel('Error Rate');
    title(sprintf('Depolarizing Channel Sim (GF(%d), N=%d, p=%d)', q, N, p));
    legend('Block Error Rate (FER)', 'Bit Error Rate (BER)', 'Location', 'southwest');
    axis([min(fm_list) max(fm_list) 1e-6 1]);

    fprintf('===== 流程结束 =====\n');
end