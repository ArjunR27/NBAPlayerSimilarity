import { useEffect, useState } from "react";
import axios from "axios";
import "./styles.css";
import Plot from "react-plotly.js";
import { ScatterData } from "plotly.js";

interface PlayerSimilarityProps {
  c_df2: {
    LR1: number[];
    LR2: number[];
    Cluster: number[];
    Name: string[];
  } | null;
  df_with_names: any[];
}
export default function PlayerSimilarity(props: PlayerSimilarityProps) {
  const [playerName, setPlayerName] = useState("");
  const [numberPlayers, setNumberPlayers] = useState(5);
  const [postResponse, setPostResponse] = useState<string[] | null>(null);
  const [melted_df, setMelted_df] = useState<{
    Value: number[];
    Stat: string[];
    Cluster: number[];
    Name: string[];
  } | null>(null);

  useEffect(() => {
    setPostResponse(null);
  }, []);

  const handleButtonClick = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        playerName,
        numberPlayers,
        c_df2: props.c_df2,
        df_with_names: props.df_with_names,
      });
      setPostResponse(response.data[0]);
      setMelted_df(response.data[1]);
      console.log(response.data);
    } catch (e) {
      console.log(e);
    }
  };
  const handleNumberChange = (value: string) => {
    const re = /^[0-9\b]+$/;
    if (value === "" || re.test(value)) {
      setNumberPlayers(Number(value));
    }
  };

  function generateColorMapping(names: string[]) {
    const uniqueNames = [...new Set(names)];
    const colors = uniqueNames.map(
      (_, index) => `hsl(${(index * 360) / uniqueNames.length}, 70%, 50%)`
    );
    return uniqueNames.reduce((acc, name, index) => {
      acc[name] = colors[index];
      return acc;
    }, {} as Record<string, string>);
  }

  const plotData: Partial<ScatterData>[] =
    melted_df != null
      ? (() => {
          // Generate color mapping for players
          const nameToColor = generateColorMapping(melted_df["Name"]);
          const colors = melted_df["Name"].map((name) => nameToColor[name]);

          return [
            {
              x: melted_df["Stat"],
              y: melted_df["Value"],
              text: melted_df["Name"], // Add hover names
              type: "scatter",
              mode: "markers",
              marker: {
                size: 10,
                color: colors, // Use the mapped colors
              },
            },
          ];
        })()
      : [];

  // Layout configuration
  const layout = {
    title: `Detailed Stats for Similar Players`,
    xaxis: { title: "" },
    yaxis: { title: "" },
    hovermode: "closest" as "closest",
    autosize: true,
  };

  return (
    props.c_df2 != null && (
      <div className="player-similarity">
        <p className="player-similarity-text">
          Nice! Above is a 2d visual representation of the clusters we created!
          Different clusters are grouped by color, and you can hover your mouse
          over a data point to view the player name. You can also query our
          results to get the closest n players to a specific player. Below, set
          your desired player, and the number of closest players to that person!
        </p>
        <div className="find-bar">
          <p>Find </p>
          <input
            defaultValue={5}
            value={numberPlayers}
            onChange={(e) => handleNumberChange(e.target.value)}
            className="find-number"
          />
          <p> similar players to</p>

          <input
            placeholder="player name"
            onChange={(e) => setPlayerName(e.target.value)}
            className="find-player"
          />
          <button className="find-button" onClick={handleButtonClick}>
            go
          </button>
        </div>
        {postResponse !== null && postResponse.length === 0 ? (
          <p style={{ textAlign: "center" }}>Player not found</p>
        ) : postResponse !== null && postResponse.length > 0 ? (
          <>
            {postResponse.map((player) => (
              <p style={{ textAlign: "center" }} key={player}>
                {player}
              </p>
            ))}
            <p style={{ textAlign: "center", width: "75%" }}>
              Cool! Above is the list of similar players, and below is a graph
              of each player's specific stats. You can see how they're all
              pretty close!
            </p>
            <Plot
              data={plotData}
              layout={layout}
              style={{ width: "100%" }}
              useResizeHandler={true}
            />
          </>
        ) : null}
      </div>
    )
  );
}
