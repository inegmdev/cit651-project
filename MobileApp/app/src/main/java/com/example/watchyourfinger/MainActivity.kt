package com.example.watchyourfinger

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.android.volley.Request
import com.android.volley.toolbox.JsonObjectRequest
import com.android.volley.toolbox.Volley
import com.example.watchyourfinger.model.PhrasePrediction
import com.example.watchyourfinger.ui.theme.WatchYourFingerTheme
import com.example.watchyourfinger.ui.theme.drawerShape
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val context = LocalContext.current
            var phrase by remember { mutableStateOf(getString(R.string.phrase)) }
            val response = remember { mutableStateOf("no response") }
            val queue = Volley.newRequestQueue(context)
            var url by remember { mutableStateOf("http://192.168.100.117:5000/predict/") }
            val scaffoldState = rememberScaffoldState()
            val scope = rememberCoroutineScope()
            val predictionsList = remember { mutableStateListOf<PhrasePrediction>() }
            WatchYourFingerTheme {
                // A surface container using the 'background' color from the theme
//                Surface(color = MaterialTheme.colors.background) {
//                    Greeting("Android")
//                }
                fun toggleDrawer(){

                    scope.launch {
                        scaffoldState.drawerState.apply {
                            if (isClosed) open() else close()
                        }
                    }
                }
                Scaffold(
                    scaffoldState = scaffoldState,
                    drawerContent = {

                        Drawer( onToggleDrawer = { toggleDrawer ()})

                    },
                    drawerShape = drawerShape,
                    topBar = {
                        TopBarWithDrawer(
                            title = stringResource(id = R.string.app_name),
                            onToggleDrawer = { toggleDrawer() })
                    }

                )
                {
                    Column(
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.padding(8.dp)
                    ) {


                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {

                            OutlinedTextField(
                                value = url,
                                onValueChange = { url = it },
                                label = { Text(text = "URL") },
                                maxLines = 1,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(0.dp),

                            )

                        }
//                        Spacer(modifier = Modifier.height(10.dp))
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(75.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            OutlinedTextField(
                                value = phrase,
                                onValueChange = { phrase = it },
                                label = { Text(text = "Phrase") },
                                maxLines = 2,
                                modifier = Modifier
                                    .fillMaxWidth(0.75F)
                                    .fillMaxHeight()

                            )
                            Button(
                                onClick = {
                                    val newUrl = "$url$phrase"
                                    val jsonObjectRequest =
                                        JsonObjectRequest(Request.Method.GET, newUrl, null,
                                            { r ->
                                                val pp = PhrasePrediction(
                                                    r.getString("phrase"),
                                                    r.getString("prediction")
                                                )
                                                response.value =
                                                    "Response: %s".format(pp.toString())
                                                predictionsList.add(pp)
                                            },
                                            { e ->
                                                response.value = "That didn't work! $e"
                                            }
                                        )

                                    println("Sending request")
//                                    response.value = "Sending request"


                                    queue.add(jsonObjectRequest)
                                },
                                modifier = Modifier
                                    .padding(top = 8.dp)
                                    .fillMaxSize()
                            ) {
                                Text(text = "Send")
                            }
                        }

                        LazyColumn() {
                            items(predictionsList) { invoice ->
                                PredictionCard(invoice)
                                Spacer(modifier = Modifier.height(10.dp))
                            }


                        }
                        //}
                    }
                }
            }
        }
    }
}


@Composable
fun PredictionCard(prediction: PhrasePrediction) {
    Card(
        elevation = 8.dp,
        shape = RoundedCornerShape(16.dp)
    ) {

        Column(
            verticalArrangement = Arrangement.Center,

        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth(), verticalAlignment = Alignment.CenterVertically
            ) {

                Row(
                    modifier = Modifier
                        .fillMaxWidth(0.9F)
                        .padding(10.dp),

                ) {
                    Text(
                        text = prediction.phrase,
                        style = MaterialTheme.typography.body1,
                        maxLines = 2
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(4.dp),
                    horizontalArrangement = Arrangement.spacedBy(0.dp, Alignment.End)
                ) {
                    RoundTag(
                        color = getColor(prediction.prediction),
                        shortText = getAbbrev(prediction.prediction)
                    )
                }

            }


        }

//        }
    }
}
