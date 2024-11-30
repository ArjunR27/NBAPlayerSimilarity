import axios from "axios";
import { ScatterData } from "plotly.js";
import { useEffect, useState } from "react";
import Plot from "react-plotly.js";

interface PlayerGraphProps {
  c_df2: {
    LR1: number[];
    LR2: number[];
    Cluster: number[];
    Name: string[];
  } | null;
  df_with_names: any[];
  setC_df2: React.Dispatch<
    React.SetStateAction<{
      LR1: number[];
      LR2: number[];
      Cluster: number[];
      Name: string[];
    } | null>
  >;
  setDf_with_names: React.Dispatch<React.SetStateAction<any[]>>;
}

export default function PlayerGraph(props: PlayerGraphProps) {
  const [year, setYear] = useState<number | null>(null);
  const [inputYear, setInputYear] = useState("");

  const handleYearChange = (value: string) => {
    const re = /^[0-9\b]+$/;
    if (value === "" || re.test(value)) {
      setInputYear(value);
    }
  };

  const handleButtonClick = () => {
    setYear(Number(inputYear));
  };

  const sendRequest = async () => {
    if (year === null || year === 0) {
      props.setC_df2(null);
    } else {
      try {
        const response = await axios.post("http://127.0.0.1:5000/build", {
          year,
        });
        console.log(response);
        console.log(response.data[0]);
        props.setC_df2(response.data[0]);
        props.setDf_with_names(response.data[1]);
      } catch (e) {
        console.log(e);
      }
    }
  };
  useEffect(() => {
    sendRequest();
  }, [year]);

  const plotData: Partial<ScatterData>[] =
    props.c_df2 != null
      ? [
          {
            x: props.c_df2["LR1"],
            y: props.c_df2["LR2"],
            text: props.c_df2["Name"], // Add hover names
            type: "scatter",
            mode: "markers",
            marker: {
              size: 10,
              color: props.c_df2["Cluster"], // Use Cluster for color
              colorscale: "Viridis", // Customize colorscale
            },
          },
        ]
      : [];

  // Layout configuration
  const layout = {
    title: `Clusters for NBA ${year} Season`,
    xaxis: { title: "" },
    yaxis: { title: "" },
    hovermode: "closest" as "closest",
    autosize: true,
  };
  return (
    <div className="player-graph">
      <div className="player-graph-text">
        <p style={{ textAlign: "center" }}>
          Welcome to our website! We used an autoencode and k-means clustering
          to group together NBA players based on season stats. Here, you can
          enter a NBA season year between 2000 and 2024, and we'll show you the
          clusters we created!
        </p>
        <div className="player-graph-text-request">
          <input
            value={inputYear}
            className="player-graph-text-input"
            placeholder="enter year here..."
            onChange={(e) => handleYearChange(e.target.value)}
          ></input>
          <button
            className="player-graph-text-button"
            onClick={handleButtonClick}
          >
            go
          </button>
        </div>
      </div>
      {props.c_df2 !== null && props.c_df2 !== undefined && (
        <Plot
          data={plotData}
          layout={layout}
          style={{ width: "50%" }}
          useResizeHandler={true}
        />
      )}
    </div>
  );
}
