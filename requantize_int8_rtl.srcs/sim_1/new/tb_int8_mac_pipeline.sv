`timescale 1ns/1ps
module tb_int8_mac_pipeline;
localparam int INPUT_WIDTH = 8;
localparam int ACC_WIDTH = 32;
localparam int EXPECTED_LINE_COUNT = 202;
localparam string VECTOR_FILE = "F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/mac_control_vectors.txt";
logic clk;
logic rst_n;
logic clear;
logic enable;
logic in_valid;
logic signed [INPUT_WIDTH-1:0] activation;
logic signed [INPUT_WIDTH-1:0] weight;
logic out_valid;
logic signed [ACC_WIDTH-1:0] acc_out;
integer vector_file;
integer scan_count;
integer header_result;
integer line_count;
integer valid_output_count;
integer mismatch_count;
integer clear_file;
integer enable_file;
integer in_valid_file;
integer activation_file;
integer weight_file;
integer expected_out_valid_file;
integer signed expected_acc_file;
reg [8*256-1:0] header_line;
int8_mac_pipeline #(
    .INPUT_WIDTH(INPUT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .clear(clear),
    .enable(enable),
    .in_valid(in_valid),
    .activation(activation),
    .weight(weight),
    .out_valid(out_valid),
    .acc_out(acc_out)
);
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end
initial begin
    rst_n = 1'b0;
    clear = 1'b0;
    enable = 1'b0;
    in_valid = 1'b0;
    activation = '0;
    weight = '0;
    line_count = 0;
    valid_output_count = 0;
    mismatch_count = 0;
    vector_file = $fopen(VECTOR_FILE, "r");
    if (vector_file == 0) begin
        $display("ERROR: cannot open vector file");
        $display("FILE: %s", VECTOR_FILE);
        $fatal;
    end
    header_result = $fgets(header_line, vector_file);
    $display("Opened vector file successfully");
    $display("Vector file: %s", VECTOR_FILE);
    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;
    while (!$feof(vector_file)) begin
        scan_count = $fscanf(
            vector_file,
            "%d %d %d %d %d %d %d\n",
            clear_file,
            enable_file,
            in_valid_file,
            activation_file,
            weight_file,
            expected_out_valid_file,
            expected_acc_file
        );
        if (scan_count == 7) begin
            @(negedge clk);
            clear = clear_file[0];
            enable = enable_file[0];
            in_valid = in_valid_file[0];
            activation = activation_file;
            weight = weight_file;
            @(posedge clk);
            #1;
            line_count = line_count + 1;
            if (expected_out_valid_file != 0) begin
                valid_output_count = valid_output_count + 1;
            end
            if ((out_valid !== expected_out_valid_file[0]) || ($signed(acc_out) !== expected_acc_file)) begin
                mismatch_count = mismatch_count + 1;
                $display("FAIL line=%0d clear=%0d enable=%0d in_valid=%0d activation=%0d weight=%0d rtl_out_valid=%0d expected_out_valid=%0d rtl_acc=%0d expected_acc=%0d", line_count, clear_file, enable_file, in_valid_file, activation_file, weight_file, out_valid, expected_out_valid_file, $signed(acc_out), expected_acc_file);
            end else begin
                $display("PASS line=%0d clear=%0d enable=%0d in_valid=%0d out_valid=%0d acc_out=%0d", line_count, clear_file, enable_file, in_valid_file, out_valid, $signed(acc_out));
            end
        end
    end
    $fclose(vector_file);
    @(negedge clk);
    clear = 1'b0;
    enable = 1'b0;
    in_valid = 1'b0;
    activation = '0;
    weight = '0;
    #20;
    $display("--------------------------------------------------");
    $display("line_count         = %0d", line_count);
    $display("valid_output_count = %0d", valid_output_count);
    $display("mismatch_count     = %0d", mismatch_count);
    if ((line_count == EXPECTED_LINE_COUNT) && (mismatch_count == 0)) begin
        $display("TEST RESULT: PASS");
    end else begin
        $display("TEST RESULT: FAIL");
    end
    $display("--------------------------------------------------");
    $finish;
end
endmodule