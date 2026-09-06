// ============================================================================
// 文件: tb_int8_attn_v.sv
// 阶段: D10
// 作用: 对拍 Attention Probability @ V（Y = P @ V）
// DUT: int8_attn_v
// 向量: attn_v_vectors.txt（seed=20260903，5 case）
// CLI: scripts/run_attn_v_xsim.bat
// ============================================================================
//
// 流程: send_cmd(seq) → stream P/V → receive Y[row][col] INT8 对拍
//
`timescale 1ns/1ps
module tb_int8_attn_v;
    localparam int INPUT_WIDTH=8, P_WIDTH=16, MAX_SEQ=4, HEAD_DIM=16;
    localparam int SEQ_WIDTH=$clog2(MAX_SEQ+1), D_WIDTH=$clog2(HEAD_DIM+1);
    localparam int MAX_P=MAX_SEQ*MAX_SEQ, MAX_V=MAX_SEQ*HEAD_DIM;
    localparam int EXPECTED_NUM_CASES=5, EXPECTED_SEED=20260903;
    localparam string VECTOR_FILE="F:/Users/22563/Desktop/FPGAlearn/requantize_int8_rtl/attn_v_vectors.txt";

    logic clk, rst_n, cmd_valid, cmd_ready;
    logic [SEQ_WIDTH-1:0] cmd_seq;
    logic p_valid, p_ready; logic [P_WIDTH-1:0] p_data; logic [SEQ_WIDTH-1:0] p_row, p_col;
    logic v_valid, v_ready; logic signed [INPUT_WIDTH-1:0] v_data;
    logic y_valid, y_ready; logic signed [INPUT_WIDTH-1:0] y_data;
    logic [SEQ_WIDTH-1:0] y_row; logic [D_WIDTH-1:0] y_col;

    integer vf, test_count, mismatch_count, header_seed, header_cases;
    string line_str;

    int8_attn_v dut(.*);

    task automatic send_cmd(input integer s);
        begin @(negedge clk); cmd_seq=s[SEQ_WIDTH-1:0]; cmd_valid=1;
            while(!cmd_ready) @(negedge clk); @(posedge clk); @(negedge clk); cmd_valid=0; end
    endtask
    task automatic send_p(input integer val, input integer r, input integer c);
        begin @(negedge clk); p_data=val[P_WIDTH-1:0]; p_row=r[SEQ_WIDTH-1:0]; p_col=c[SEQ_WIDTH-1:0]; p_valid=1;
            while(!p_ready) @(negedge clk); @(posedge clk); @(negedge clk); p_valid=0; end
    endtask
    task automatic send_v(input integer signed val);
        begin @(negedge clk); v_data=val; v_valid=1;
            while(!v_ready) @(negedge clk); @(posedge clk); @(negedge clk); v_valid=0; end
    endtask
    task automatic recv_y(input integer er, input integer ec, input integer signed expv, input integer cid, input integer hold);
        integer h, held, hr, hc;
        begin y_ready=0; while(!y_valid) @(negedge clk);
            held=$signed(y_data); hr=y_row; hc=y_col;
            if(hr!=er||hc!=ec||held!==expv) begin mismatch_count++; $display("FAIL c=%0d Y[%0d][%0d] rtl=%0d exp=%0d",cid,er,ec,held,expv); end
            else $display("PASS c=%0d Y[%0d][%0d]=%0d",cid,er,ec,held);
            for(h=0;h<hold;h++) begin @(negedge clk); if(!y_valid||$signed(y_data)!==held) begin mismatch_count++; end end
            y_ready=1; @(posedge clk); @(negedge clk); y_ready=0;
        end
    endtask

    function automatic bit is_marker(input string line);
        is_marker=(line=="P\n")||(line=="V\n")||(line=="EXPECT\n")||(line=="END\n")||(line=="END");
    endfunction

    task automatic read_block(input integer nexp, output integer signed vals[0:MAX_V-1], output integer nact, output string next_line, input string name);
        integer sc; integer signed v;
        begin nact=0; next_line="";
            while(!$feof(vf)) begin
                if($fgets(line_str,vf)==0) continue;
                if(line_str.len()==0) continue;
                if(line_str[0]=="#") continue;
                if(is_marker(line_str)) begin next_line=line_str; break; end
                sc=$sscanf(line_str,"%d",v); if(sc!=1) begin mismatch_count++; return; end
                if(nact>=nexp) begin mismatch_count++; return; end
                vals[nact]=v; nact++;
            end
            if(nact!=nexp) begin mismatch_count++; $display("FAIL %s count %0d/%0d",name,nact,nexp); end
        end
    endtask

    task automatic run_case(input integer cid, input integer seq_v, input integer hold);
        integer pn,vn,en,i,r,c; integer signed pv[0:MAX_V-1], vv[0:MAX_V-1], ev[0:MAX_V-1]; string marker;
        begin
            $display("CASE %0d seq=%0d",cid,seq_v);
            read_block(seq_v*seq_v,pv,pn,marker,"P"); if(pn!=seq_v*seq_v) begin test_count++; return; end
            if(marker!="V\n") begin mismatch_count++; test_count++; return; end
            read_block(seq_v*HEAD_DIM,vv,vn,marker,"V"); if(vn!=seq_v*HEAD_DIM) begin test_count++; return; end
            if(marker!="EXPECT\n") begin mismatch_count++; test_count++; return; end
            read_block(seq_v*HEAD_DIM,ev,en,marker,"EXPECT"); if(en!=seq_v*HEAD_DIM) begin test_count++; return; end
            send_cmd(seq_v);
            for(r=0;r<seq_v;r++) for(c=0;c<seq_v;c++) send_p(pv[r*seq_v+c],r,c);
            for(i=0;i<seq_v*HEAD_DIM;i++) send_v(vv[i]);
            for(r=0;r<seq_v;r++) for(c=0;c<HEAD_DIM;c++) recv_y(r,c,ev[r*HEAD_DIM+c],cid,hold);
            test_count++;
        end
    endtask

    initial begin clk=0; forever #5 clk=~clk; end
    initial begin
        integer sc,cid,seq_v,hold;
        rst_n=0; cmd_valid=0; cmd_seq=1; p_valid=0; v_valid=0; y_ready=0;
        test_count=0; mismatch_count=0; header_seed=-1; header_cases=-1;
        vf=$fopen(VECTOR_FILE,"r"); if(!vf) $fatal;
        $display("D10 Attn@V regression");
        repeat(4) @(posedge clk); @(negedge clk); rst_n=1;
        while(!$feof(vf)) begin
            if($fgets(line_str,vf)==0) continue;
            if(line_str.len()==0) continue;
            if(line_str[0]=="#") begin sc=$sscanf(line_str,"# seed=%d num_cases=%d",header_seed,header_cases); continue; end
            if(line_str.substr(0,3)=="CASE") begin
                sc=$sscanf(line_str,"CASE %d %d %d",cid,seq_v,hold);
                if(sc!=3) begin mismatch_count++; continue; end
                if($fgets(line_str,vf)==0||line_str!="P\n") begin mismatch_count++; continue; end
                run_case(cid,seq_v,hold);
            end
        end
        $fclose(vf);
        $display("seed=%0d test_count=%0d mismatch_count=%0d",header_seed,test_count,mismatch_count);
        if(header_seed==EXPECTED_SEED&&test_count==EXPECTED_NUM_CASES&&header_cases==EXPECTED_NUM_CASES&&mismatch_count==0) begin
            $display("TEST RESULT: PASS"); $display("PYTHON AND RTL ARE %0d/%0d EXACT.",test_count,EXPECTED_NUM_CASES);
        end else $display("TEST RESULT: FAIL");
        $finish;
    end
    initial begin #1000000000; $display("TIMEOUT"); $finish; end
endmodule
