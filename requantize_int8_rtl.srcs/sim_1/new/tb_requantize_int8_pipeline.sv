`timescale 1ns / 1ps
module tb_requantize_int8_pipeline;
logic clk;
logic rst_n;
logic in_valid;
logic signed [31:0] acc_i;
logic        [17:0] multiplier_i;
logic out_valid;
logic signed [7:0] out_i;
logic signed [31:0] accumulator_vectors [0:11];
logic        [17:0] multiplier_vectors [0:11];
logic signed  [7:0] expected_vectors [0:11];
logic [2:0] valid_history;
integer received_count;
integer mismatch_count;
integer latency_error_count;
integer i;
requantize_int8_pipeline #(
    .SHIFT_BITS(24)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .acc_i(acc_i),
    .multiplier_i(multiplier_i),
    .out_valid(out_valid),
    .out_i(out_i)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
always_ff @(posedge clk) begin
    if (!rst_n)
        valid_history <= 3'b000;
    else
        valid_history <= {valid_history[1:0], in_valid};
end
always @(negedge clk) begin
    if (rst_n) begin
        if (out_valid !== valid_history[2]) begin
            latency_error_count = latency_error_count + 1;
            $error(
                "VALID LATENCY ERROR: out_valid=%0b expected_valid=%0b",
                out_valid,
                valid_history[2]
            );
        end
        if (out_valid) begin
            if (received_count >= 12) begin
                mismatch_count = mismatch_count + 1;
                $error("EXTRA OUTPUT: rtl=%0d", $signed(out_i));
            end
            else if ($signed(out_i) === expected_vectors[received_count]) begin
                $display(
                    "PIPELINE MATCH [%0d]: rtl=%0d expected=%0d pre_clip=%0d",
                    received_count,
                    $signed(out_i),
                    expected_vectors[received_count],
                    $signed(dut.shifted_s2)
                );
            end
            else begin
                mismatch_count = mismatch_count + 1;
                $error(
                    "PIPELINE MISMATCH [%0d]: rtl=%0d expected=%0d pre_clip=%0d",
                    received_count,
                    $signed(out_i),
                    expected_vectors[received_count],
                    $signed(dut.shifted_s2)
                );
            end
            received_count = received_count + 1;
        end
    end
end
initial begin
    accumulator_vectors[0]  = 32'sd100;
    accumulator_vectors[1]  = 32'sd23673;
    accumulator_vectors[2]  = -32'sd10447;
    accumulator_vectors[3]  = 32'sd24709;
    accumulator_vectors[4]  = 32'sd11295;
    accumulator_vectors[5]  = 32'sd1755;
    accumulator_vectors[6]  = -32'sd3961;
    accumulator_vectors[7]  = 32'sd3189;
    accumulator_vectors[8]  = 32'sd11428;
    accumulator_vectors[9]  = 32'sd4030;
    accumulator_vectors[10] = 32'sd281;
    accumulator_vectors[11] = 32'sd2340;
    multiplier_vectors[0]  = 18'd26979;
    multiplier_vectors[1]  = 18'd88887;
    multiplier_vectors[2]  = 18'd205831;
    multiplier_vectors[3]  = 18'd45891;
    multiplier_vectors[4]  = 18'd26979;
    multiplier_vectors[5]  = 18'd88887;
    multiplier_vectors[6]  = 18'd205831;
    multiplier_vectors[7]  = 18'd45891;
    multiplier_vectors[8]  = 18'd26979;
    multiplier_vectors[9]  = 18'd88887;
    multiplier_vectors[10] = 18'd205831;
    multiplier_vectors[11] = 18'd45891;
    expected_vectors[0]  = 8'sd0;
    expected_vectors[1]  = 8'sd125;
    expected_vectors[2]  = -8'sd127;
    expected_vectors[3]  = 8'sd68;
    expected_vectors[4]  = 8'sd18;
    expected_vectors[5]  = 8'sd9;
    expected_vectors[6]  = -8'sd49;
    expected_vectors[7]  = 8'sd9;
    expected_vectors[8]  = 8'sd18;
    expected_vectors[9]  = 8'sd21;
    expected_vectors[10] = 8'sd3;
    expected_vectors[11] = 8'sd6;
    rst_n = 1'b0;
    in_valid = 1'b0;
    acc_i = 32'sd0;
    multiplier_i = 18'd0;
    received_count = 0;
    mismatch_count = 0;
    latency_error_count = 0;
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    for (i = 0; i < 12; i = i + 1) begin
        in_valid = 1'b1;
        acc_i = accumulator_vectors[i];
        multiplier_i = multiplier_vectors[i];
        @(negedge clk);
    end
    in_valid = 1'b0;
    acc_i = 32'sd0;
    multiplier_i = 18'd0;
    wait (received_count == 12);
    repeat (2) @(posedge clk);
    $display("============================================================");
    $display("Pipeline comparison summary");
    $display("received_count      = %0d", received_count);
    $display("mismatch_count      = %0d", mismatch_count);
    $display("latency_error_count = %0d", latency_error_count);
    $display("============================================================");
    if (mismatch_count == 0 && latency_error_count == 0) begin
        $display("PIPELINE RTL AND PYTHON V2 ARE 12/12 EXACT.");
        $display("PIPELINE VALID LATENCY CHECK PASSED.");
        $finish;
    end
    else begin
        $fatal(
            1,
            "PIPELINE TEST FAILED: mismatch=%0d latency_error=%0d",
            mismatch_count,
            latency_error_count
        );
    end
end
endmodule