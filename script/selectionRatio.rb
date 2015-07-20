#!/usr/ruby

sRatio = [ 0.5, 1, 3, 8, 15, 20 ]

cnt = 0 # index for selectionRatio array
adjustment = 0.1
output = ""


while cnt < sRatio.length do
	for i in 1...9 # digit place 
		for j in 1...9 # weigh of each digit
			puts "i " +i.to_s+", j "+j.to_s+"\n"
			puts "adjustment : "+adjustment.to_s
			w = 1.0 / 10.0**i

			command = "cuda 1000000 1 " + adjustment.to_s + " > selectionRatioRUBY"
			#execute the command
			system(command)

			#open the file and read output
			File.open("out") do |file|
				line = file.gets
				output = line
				output = (output.to_f*10.0).to_i
				output = output.to_f*0.1
			#file end
			end

			puts "current output : "+output.to_s
			puts "selectionRatio : "+sRatio[cnt].to_s

			#compare selectionRatio and update
			#found 
			if output.to_f == sRatio[cnt].to_f
				puts "Find the adjustment\n"
				#next selectionRatio
				cnt += 1
				#print the result
				puts "adjustment : "+adjustment.to_s
				adjustment = 0.1
				puts ""
				puts ""
				break
			elsif output.to_f > sRatio[cnt].to_f
#puts "decrease the adjustment\n"
				adjustment = adjustment-w		
				adjustment = adjustment+( 1.0 / 10.0**(i+1))
				break
			#update adjustment 
			else
#puts "increase the adjustment\n"
				adjustment = adjustment+w
			end
		end
		if adjustment.to_f == 0.1
			break
			end
	end
end

			system("rm -rf selectionRatioRUBY")
