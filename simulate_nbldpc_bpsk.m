function simulate_nbldpc_bpsk()
    addpath('NB_LDPC/BP-decoder-for-NB_LDPC-codes-master');

    % 配置
    alist_file = fullfile(pwd, 'NB_LDPC', 'NB_LDPC_FB_public-master', 'matrices', 'H_gamma_nb.alist');
    q = 64;                  % GF 阶 (需与 alist 一致; 若不一致需重新 mex -DLOG_Q=6)
    max_iter = 50;           % 译码最大迭代
    EbN0_dB_list = 0:1:7;    % 扫描 SNR
    max_frames = 2000;       % 每个 SNR 最多仿真帧数
    min_err_frames = 50;     % 每个 SNR 至少统计的误帧数（提前停止阈值）

    % 初始化解码器
    [decoder_handle, N, M] = decode_soft(0, alist_file);

    % 码率估计：R ≈ 1 - M/N（常用近似）
    R = max(1 - M / N, 1e-3);

    % 常量
    m_bits = log2(q);        % 每符号比特数

    BER = zeros(size(EbN0_dB_list));
    FER = zeros(size(EbN0_dB_list));

    fprintf('Start NB-LDPC BPSK AWGN simulation: N=%d, M=%d, R≈%.4f, GF(%d)\n', N, M, R, q);

    for idx = 1:numel(EbN0_dB_list)
        EbN0_dB = EbN0_dB_list(idx);
        EbN0 = 10^(EbN0_dB/10);
        N0 = 1/(R * EbN0);
        sigma2 = N0/2;
        sigma = sqrt(sigma2);

        total_bit_errors = 0;
        total_symbol_errors = 0;
        total_frames = 0;

        while total_frames < max_frames && total_symbol_errors < min_err_frames
            total_frames = total_frames + 1;

            % 发送全零码字（N 个非二元符号 → m_bits x N 个比特全 0）
            bits_tx = zeros(m_bits, N);

            % BPSK 调制 + AWGN
            bpsk_tx = 1 - 2*bits_tx;
            rx = bpsk_tx + sigma*randn(m_bits, N);

            % 逐比特 LLR：L = 2*y/sigma^2
            bit_llr = 2*rx./sigma2;

            % 组装 QxN 的符号对数似然 (逐候选符号累加其比特对应的 log 概率)
            llrs = zeros(q, N);
            for s = 0:q-1
                bits_s = bitget(s, m_bits:-1:1);
                pattern = repmat(bits_s.', 1, N);
                sym_logp = ((1 - pattern).* (bit_llr/2)) + (pattern.* (-bit_llr/2));
                llrs(s+1, :) = sum(sym_logp, 1);
            end

            % BP 译码
            [ok, iters, hard_sym, ~] = decode_soft(2, decoder_handle, llrs, max_iter); %#ok<ASGLU>

            % 统计误差（与全零符号比较）
            sym_errs = sum(hard_sym ~= 0);
            if sym_errs > 0
                % 符号->比特
                hard_bits = zeros(m_bits, N);
                for n = 1:N
                    s = hard_sym(n);
                    hard_bits(:, n) = bitget(s, m_bits:-1:1).';
                end
                bit_errs = sum(hard_bits(:) ~= 0);
                total_symbol_errors = total_symbol_errors + 1;
                total_bit_errors = total_bit_errors + bit_errs;
            end

            if mod(total_frames, 100) == 0
                fprintf('Eb/N0=%.1f dB: frames=%d, frame_err=%d\n', EbN0_dB, total_frames, total_symbol_errors);
            end
        end

        num_bits_per_frame = N * m_bits;
        BER(idx) = total_bit_errors / max(total_frames * num_bits_per_frame, 1);
        FER(idx) = total_symbol_errors / max(total_frames, 1);

        fprintf('SNR=%.1f dB -> BER=%.3e, FER=%.3e (frames=%d)\n', EbN0_dB, BER(idx), FER(idx), total_frames);
    end

    % 绘图（论文风格）
    figure; grid on; hold on;
    semilogy(EbN0_dB_list, BER, 'ro-','LineWidth', 2, 'MarkerSize', 7);
    semilogy(EbN0_dB_list, FER, 'bs--','LineWidth', 2, 'MarkerSize', 7);
    xlabel('E_b/N_0 (dB)'); ylabel('Error Rate');
    title(sprintf('NB-LDPC over BPSK AWGN (GF(%d)), N=%d, R\\approx%.3f, iter=%d', q, N, R, max_iter));
    legend('BER','FER','Location','southwest');
    axis([min(EbN0_dB_list) max(EbN0_dB_list) 1e-6 1]);
end