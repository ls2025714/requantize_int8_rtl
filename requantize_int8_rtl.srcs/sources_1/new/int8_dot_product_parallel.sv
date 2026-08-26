module int8_dot_product_parallel #(
    parameter int INPUT_WIDTH     = 8,
    parameter int PRODUCT_WIDTH   = 16,
    parameter int PAIR_WIDTH      = 17,
    parameter int PARTIAL_WIDTH   = 18,
    parameter int ACC_WIDTH       = 32,
    parameter int PARALLELISM     = 4,
    parameter int MAX_K           = 256,
    parameter int LENGTH_WIDTH    = $clog2(MAX_K + 1),
    parameter int PARTIAL_LATENCY = 2,
    parameter int ACC_LATENCY     = 3
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          cmd_valid,
    output logic                          cmd_ready,
    input  logic [LENGTH_WIDTH-1:0]       cmd_length,
    input  logic                          s_valid,
    output logic                          s_ready,
    input  logic signed [INPUT_WIDTH-1:0] s_a0,
    input  logic signed [INPUT_WIDTH-1:0] s_a1,
    input  logic signed [INPUT_WIDTH-1:0] s_a2,
    input  logic signed [INPUT_WIDTH-1:0] s_a3,
    input  logic signed [INPUT_WIDTH-1:0] s_b0,
    input  logic signed [INPUT_WIDTH-1:0] s_b1,
    input  logic signed [INPUT_WIDTH-1:0] s_b2,
    input  logic signed [INPUT_WIDTH-1:0] s_b3,
    input  logic [PARALLELISM-1:0]        s_keep,
    input  logic                          acc_clear,
    input  logic                          acc_enable,
    output logic                          m_valid,
    input  logic                          m_ready,
    output logic signed [ACC_WIDTH-1:0]   m_result,
    output logic                          partial_valid,
    output logic signed [PARTIAL_WIDTH-1:0] partial_sum,
    output logic                          acc_valid,
    output logic signed [ACC_WIDTH-1:0]   acc_value
);
    // Stage 1: 4-lane products (16-bit)
    logic signed [PRODUCT_WIDTH-1:0] product0_s1;
    logic signed [PRODUCT_WIDTH-1:0] product1_s1;
    logic signed [PRODUCT_WIDTH-1:0] product2_s1;
    logic signed [PRODUCT_WIDTH-1:0] product3_s1;
    logic                            valid_s1;

    // Stage 2: pair sums (17-bit)
    logic signed [PAIR_WIDTH-1:0] pair_sum0_s2;
    logic signed [PAIR_WIDTH-1:0] pair_sum1_s2;
    logic                         valid_s2;

    // Stage 3: partial sum (18-bit)
    logic signed [PARTIAL_WIDTH-1:0] partial_sum_s3;
    logic                            valid_s3;

    // Stage 4: INT32 accumulation
    logic signed [ACC_WIDTH-1:0] acc_reg;
    logic                        valid_s4;

    logic input_accept;
    logic signed [PRODUCT_WIDTH-1:0] product0_in;
    logic signed [PRODUCT_WIDTH-1:0] product1_in;
    logic signed [PRODUCT_WIDTH-1:0] product2_in;
    logic signed [PRODUCT_WIDTH-1:0] product3_in;
    logic signed [PAIR_WIDTH-1:0]    product0_ext17;
    logic signed [PAIR_WIDTH-1:0]    product1_ext17;
    logic signed [PAIR_WIDTH-1:0]    product2_ext17;
    logic signed [PAIR_WIDTH-1:0]    product3_ext17;
    logic signed [PAIR_WIDTH-1:0]    pair_sum0_comb;
    logic signed [PAIR_WIDTH-1:0]    pair_sum1_comb;
    logic signed [PARTIAL_WIDTH-1:0] pair_sum0_ext18;
    logic signed [PARTIAL_WIDTH-1:0] pair_sum1_ext18;
    logic signed [PARTIAL_WIDTH-1:0] partial_sum_comb;

    assign input_accept = s_valid && s_ready;
    // Task B: always ready so continuous 1 beat/cycle is possible after fill.
    // Task C will gate s_ready with RUN/DRAIN state.
    assign s_ready = 1'b1;

    assign product0_in = s_keep[0] ? (s_a0 * s_b0) : PRODUCT_WIDTH'(0);
    assign product1_in = s_keep[1] ? (s_a1 * s_b1) : PRODUCT_WIDTH'(0);
    assign product2_in = s_keep[2] ? (s_a2 * s_b2) : PRODUCT_WIDTH'(0);
    assign product3_in = s_keep[3] ? (s_a3 * s_b3) : PRODUCT_WIDTH'(0);

    assign product0_ext17 = {{1{product0_s1[PRODUCT_WIDTH-1]}}, product0_s1};
    assign product1_ext17 = {{1{product1_s1[PRODUCT_WIDTH-1]}}, product1_s1};
    assign product2_ext17 = {{1{product2_s1[PRODUCT_WIDTH-1]}}, product2_s1};
    assign product3_ext17 = {{1{product3_s1[PRODUCT_WIDTH-1]}}, product3_s1};
    assign pair_sum0_comb = product0_ext17 + product1_ext17;
    assign pair_sum1_comb = product2_ext17 + product3_ext17;

    assign pair_sum0_ext18 = {{1{pair_sum0_s2[PAIR_WIDTH-1]}}, pair_sum0_s2};
    assign pair_sum1_ext18 = {{1{pair_sum1_s2[PAIR_WIDTH-1]}}, pair_sum1_s2};
    assign partial_sum_comb = pair_sum0_ext18 + pair_sum1_ext18;

    assign partial_valid = valid_s3;
    assign partial_sum   = partial_sum_s3;
    assign acc_valid     = valid_s4;
    assign acc_value     = acc_reg;

    // Command / output stream reserved for Task C FSM.
    assign cmd_ready = 1'b0;
    assign m_valid   = 1'b0;
    assign m_result  = '0;

    // Stage 1: register masked products from accepted beat.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            product0_s1 <= '0;
            product1_s1 <= '0;
            product2_s1 <= '0;
            product3_s1 <= '0;
            valid_s1    <= 1'b0;
        end else if (acc_clear) begin
            valid_s1 <= 1'b0;
        end else begin
            valid_s1 <= input_accept;
            if (input_accept) begin
                product0_s1 <= product0_in;
                product1_s1 <= product1_in;
                product2_s1 <= product2_in;
                product3_s1 <= product3_in;
            end
        end
    end

    // Stage 2: register pair sums from Stage-1 products only.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pair_sum0_s2 <= '0;
            pair_sum1_s2 <= '0;
            valid_s2     <= 1'b0;
        end else if (acc_clear) begin
            valid_s2 <= 1'b0;
        end else begin
            valid_s2 <= valid_s1;
            if (valid_s1) begin
                pair_sum0_s2 <= pair_sum0_comb;
                pair_sum1_s2 <= pair_sum1_comb;
            end
        end
    end

    // Stage 3: register partial sum from Stage-2 pair sums only.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            partial_sum_s3 <= '0;
            valid_s3       <= 1'b0;
        end else if (acc_clear) begin
            valid_s3 <= 1'b0;
        end else begin
            valid_s3 <= valid_s2;
            if (valid_s2) begin
                partial_sum_s3 <= partial_sum_comb;
            end
        end
    end

    // Stage 4: accumulate Stage-3 partial sum; emit acc_valid one cycle later.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc_reg  <= '0;
            valid_s4 <= 1'b0;
        end else if (acc_clear) begin
            acc_reg  <= '0;
            valid_s4 <= 1'b0;
        end else begin
            valid_s4 <= acc_enable && valid_s3;
            if (acc_enable && valid_s3) begin
                acc_reg <= acc_reg + {{(ACC_WIDTH-PARTIAL_WIDTH){partial_sum_s3[PARTIAL_WIDTH-1]}}, partial_sum_s3};
            end
        end
    end

endmodule
